"""
main.py — FastAPI application entry point for the expense tracker.

Provides:
  • CORS for local front-end dev servers
  • User registration and login with bcrypt + JWT
  • A ``get_current_user`` dependency for protected routes
  • Startup loading of the SQLite DB and ML model into app state
  • Mounting of the ``routes.py`` router for transaction endpoints

Run locally::

    uvicorn main:app --reload --port 8000

Environment variables
---------------------
SECRET_KEY      JWT signing secret (required in production).
                Defaults to a dev-only fallback if unset.
ALGORITHM       JWT algorithm. Defaults to HS256.
TOKEN_EXPIRE_MINUTES
                Access-token lifetime in minutes. Defaults to 60.
MODEL_PATH      Path to the saved model.joblib. Defaults to model.joblib.
DATABASE_URL    SQLAlchemy URL. Defaults to sqlite:///expenses.db.
"""

from __future__ import annotations
from fastapi.staticfiles import StaticFiles
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Generator

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

# Local modules
from database import SessionLocal, User, init_db
from model_tfidf import ExpenseClassifier
from routes import router as expense_router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

# ── Config from environment ───────────────────────────────────────────────────

_DEV_SECRET = "dev-only-secret-change-in-production-please"  # noqa: S105

SECRET_KEY            : str = os.getenv("SECRET_KEY", _DEV_SECRET)
ALGORITHM             : str = os.getenv("ALGORITHM", "HS256")
TOKEN_EXPIRE_MINUTES  : int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
MODEL_PATH            : Path = Path(os.getenv("MODEL_PATH", "model.joblib"))

if SECRET_KEY == _DEV_SECRET:
    log.warning(
        "SECRET_KEY is using the insecure dev fallback. "
        "Set the SECRET_KEY environment variable before deploying."
    )

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Expense Tracker API",
    description=(
        "Classifies bank transactions with a TF-IDF + Logistic Regression model, "
        "supports user corrections, and retrains automatically via active learning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React / Next.js dev server
        "http://localhost:5500",   # VS Code Live Server / plain HTML dev
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security scheme ───────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=True)

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    """Body accepted by ``POST /auth/register``."""

    username: str = Field(..., min_length=3, max_length=64, examples=["alice"])
    email:    EmailStr = Field(..., examples=["alice@example.com"])
    password: str = Field(..., min_length=8, examples=["s3cr3tP@ss"])


class RegisterResponse(BaseModel):
    """Returned by ``POST /auth/register`` on success."""

    user_id:  int
    username: str
    token:    str


class LoginRequest(BaseModel):
    """Body accepted by ``POST /auth/login``."""

    email:    EmailStr = Field(..., examples=["alice@example.com"])
    password: str      = Field(..., examples=["s3cr3tP@ss"])


class LoginResponse(BaseModel):
    """Returned by ``POST /auth/login`` on success."""

    user_id:    int
    username:   str
    token:      str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    """Safe public representation of a ``User`` row (no password)."""

    id:         int
    username:   str
    email:      str
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """Returned by ``GET /health``."""

    status:      str
    db:          str
    model:       str
    timestamp:   datetime


# ── JWT helpers ───────────────────────────────────────────────────────────────


def _create_access_token(user_id: int, username: str) -> str:
    """
    Mint a signed JWT for the given user.

    Args:
        user_id:  Primary key from the ``users`` table.
        username: Display name embedded in the token payload.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "exp":      expire,
        "iat":      datetime.now(tz=timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    """
    Decode and verify a JWT, raising ``HTTPException`` on any failure.

    Args:
        token: Raw JWT string (without the ``Bearer `` prefix).

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException 401: If the token is invalid, expired, or missing ``sub``.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise credentials_exc
        return payload
    except JWTError:
        raise credentials_exc


# ── Password helpers ──────────────────────────────────────────────────────────


def _hash_password(plain: str) -> str:
    """Return a bcrypt hash of ``plain``."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode(), salt).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    """Return ``True`` if ``plain`` matches the bcrypt ``hashed`` string."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── DB dependency ─────────────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; close it when the request finishes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbDep = Annotated[Session, Depends(get_db)]

# ── Auth dependency ───────────────────────────────────────────────────────────


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    db: DbDep,
) -> User:
    """
    FastAPI dependency — validate the Bearer JWT and return the ``User`` row.

    Inject into any route that requires authentication::

        @app.get("/me")
        def me(user: User = Depends(get_current_user)):
            return {"id": user.id, "username": user.username}

    Args:
        credentials: Parsed ``Authorization: Bearer <token>`` header.
        db:          Injected DB session.

    Returns:
        The ``User`` ORM object for the authenticated user.

    Raises:
        HTTPException 401: Token invalid / expired.
        HTTPException 404: User no longer exists in the DB.
    """
    payload  = _decode_token(credentials.credentials)
    user_id  = int(payload["sub"])
    user     = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]

# ── Startup / shutdown ────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup() -> None:
    """
    On startup:

    1. Create all DB tables (idempotent).
    2. Load the ML model into ``app.state.classifier`` so routes can reuse
       the same object without re-loading from disk on every request.
    """
    log.info("Startup: initialising database …")
    init_db()
    log.info("Startup: database ready.")

    log.info("Startup: loading ML model from %s …", MODEL_PATH)
    if MODEL_PATH.exists():
        try:
            app.state.classifier = ExpenseClassifier.load(MODEL_PATH)
            log.info(
                "Startup: model loaded — categories: %s",
                app.state.classifier.classes_,
            )
        except Exception as exc:
            log.error("Startup: model load failed (%s). classifier set to None.", exc)
            app.state.classifier = None
    else:
        log.warning("Startup: model file not found at %s. classifier set to None.", MODEL_PATH)
        app.state.classifier = None


@app.on_event("shutdown")
async def shutdown() -> None:
    log.info("Shutdown: cleaning up.")


# ── Router ────────────────────────────────────────────────────────────────────

app.include_router(expense_router, prefix="/api", tags=["expenses"])

# Insights / analytics router
from insights import router as insights_router
app.include_router(insights_router, prefix="/api", tags=["insights"])

# ── Health ────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health(db: DbDep) -> HealthResponse:
    """
    Returns the liveness status of the API, database connection, and ML model.

    Performs a lightweight DB probe (``SELECT 1``) so that a connection-pool
    failure is surfaced immediately rather than on the first real request.
    """
    # Probe DB
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        log.error("Health check DB probe failed: %s", exc)
        db_status = f"error: {exc}"

    model_status = "loaded" if getattr(app.state, "classifier", None) is not None else "not loaded"

    return HealthResponse(
        status    = "ok" if db_status == "ok" else "degraded",
        db        = db_status,
        model     = model_status,
        timestamp = datetime.now(tz=timezone.utc),
    )


# ── Auth routes ───────────────────────────────────────────────────────────────


@app.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    tags=["auth"],
)
def register(body: RegisterRequest, db: DbDep) -> RegisterResponse:
    """
    Create a new user account.

    * Checks for duplicate ``username`` or ``email`` and returns ``409`` if
      either already exists.
    * Hashes the password with bcrypt before storing.
    * Returns a ready-to-use JWT so the client can start making requests
      immediately without a separate login call.

    Args:
        body: ``RegisterRequest`` with ``username``, ``email``, ``password``.
        db:   Injected DB session.

    Returns:
        ``RegisterResponse`` with ``user_id``, ``username``, ``token``.

    Raises:
        HTTPException 409: Username or email already taken.
    """
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{body.email}' is already registered.",
        )

    user = User(
        username        = body.username,
        email           = body.email,
        hashed_password = _hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = _create_access_token(user.id, user.username)
    log.info("New user registered: id=%d username=%s", user.id, user.username)

    return RegisterResponse(user_id=user.id, username=user.username, token=token)


@app.post(
    "/auth/login",
    response_model=LoginResponse,
    summary="Log in and receive a JWT",
    tags=["auth"],
)
def login(body: LoginRequest, db: DbDep) -> LoginResponse:
    """
    Authenticate an existing user and return a JWT.

    Returns a generic ``401`` for both *email not found* and *wrong password*
    to avoid leaking which emails are registered.

    Args:
        body: ``LoginRequest`` with ``email`` and ``password``.
        db:   Injected DB session.

    Returns:
        ``LoginResponse`` with ``user_id``, ``username``, ``token``, ``token_type``.

    Raises:
        HTTPException 401: Invalid credentials.
    """
    _invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not _verify_password(body.password, user.hashed_password):
        raise _invalid

    token = _create_access_token(user.id, user.username)
    log.info("User logged in: id=%d username=%s", user.id, user.username)

    return LoginResponse(
        user_id=user.id, username=user.username, token=token
    )


@app.get(
    "/auth/me",
    response_model=UserPublic,
    summary="Return the current authenticated user",
    tags=["auth"],
)
def me(current_user: CurrentUser) -> UserPublic:
    """
    Return the public profile of the authenticated user.

    Useful for front-ends that need to confirm token validity and display
    the user's details after a page reload.

    Args:
        current_user: Injected via ``get_current_user`` dependency.

    Returns:
        ``UserPublic`` schema (no password fields).
    """
    return UserPublic.model_validate(current_user)


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
