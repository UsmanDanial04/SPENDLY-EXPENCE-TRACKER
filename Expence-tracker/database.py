"""
database.py — SQLAlchemy / SQLite database layer for the expense tracker.
"""

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── Engine & session factory ────────────────────────────────────────────────

DATABASE_URL = "sqlite:///expenses.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Declarative base ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ──────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64),  unique=True, nullable=False, index=True)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    transactions = relationship(
        "Transaction",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class Transaction(Base):
    __tablename__ = "transactions"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(
                            Integer,
                            ForeignKey("users.id", ondelete="CASCADE"),
                            nullable=False,
                          )
    date                = Column(DateTime, nullable=False)
    description         = Column(Text, nullable=False)
    amount              = Column(Float, nullable=False)
    predicted_category  = Column(String(64),  nullable=True)
    corrected_category  = Column(String(64),  nullable=True)
    confidence_score    = Column(Float,        nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Explicit indexes for common query patterns
    __table_args__ = (
        Index("ix_transactions_user_id", "user_id"),
        Index("ix_transactions_date",    "date"),
        Index("ix_transactions_user_date", "user_id", "date"),
    )

    # Relationships
    user = relationship("User", back_populates="transactions")
    corrections = relationship(
        "ModelCorrection",
        back_populates="transaction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction id={self.id} user_id={self.user_id} "
            f"amount={self.amount} date={self.date}>"
        )


class ModelCorrection(Base):
    __tablename__ = "model_corrections"

    id                   = Column(Integer, primary_key=True, index=True)
    transaction_id       = Column(
                             Integer,
                             ForeignKey("transactions.id", ondelete="CASCADE"),
                             nullable=False,
                             index=True,
                           )
    original_prediction  = Column(String(64), nullable=False)
    corrected_label      = Column(String(64), nullable=False)
    timestamp            = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    transaction = relationship("Transaction", back_populates="corrections")

    def __repr__(self) -> str:
        return (
            f"<ModelCorrection id={self.id} "
            f"transaction_id={self.transaction_id} "
            f"{self.original_prediction!r} → {self.corrected_label!r}>"
        )


# ── Table initialisation ────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables in the database (idempotent)."""
    Base.metadata.create_all(bind=engine)


# ── FastAPI dependency ──────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session and guarantee it is closed afterwards.

    Usage in a FastAPI route::

        @router.get("/transactions")
        def list_transactions(db: Session = Depends(get_db)):
            return db.query(Transaction).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialising database …")
    init_db()

    # Quick smoke-test: inspect created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables created: {tables}")
    for table in tables:
        cols = [c["name"] for c in inspector.get_columns(table)]
        idxs = [i["name"] for i in inspector.get_indexes(table)]
        print(f"  {table}: columns={cols}")
        print(f"  {table}: indexes={idxs}")

    print("\nDone — expenses.db is ready.")
