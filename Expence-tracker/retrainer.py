"""
retrainer.py — Active learning retrainer for the expense classifier.

Monitors accumulated user corrections in the ModelCorrection table and
automatically retrains the TF-IDF + LogisticRegression pipeline when
enough corrections have been collected.

Retraining strategy
-------------------
* Original training data is loaded from ``transactions.csv``.
* Corrections are loaded from the ``ModelCorrection`` table via SQLAlchemy.
  Each correction row provides the corrected label; the corresponding
  transaction description is fetched from the ``Transaction`` table.
* Corrections are weighted 3× by tripling them in the training corpus,
  giving the model a stronger signal on recently corrected examples.
* The new model is only saved if its macro-F1 on a held-out split
  exceeds the current model's macro-F1.
* The model file is replaced atomically (write → temp file → rename)
  to avoid partial writes being read by a live API.

Usage::

    python retrainer.py          # runs __main__ simulation
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

# Local modules
from database import ModelCorrection, Transaction
from model_tfidf import CATEGORIES, ExpenseClassifier

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retrainer")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH   = Path("model.joblib")
DEFAULT_CSV_PATH     = Path("transactions.csv")
DEFAULT_LOG_PATH     = Path("retrain_log.json")
CORRECTION_WEIGHT    = 3          # duplicate corrections this many times
TEST_SPLIT_SIZE      = 0.20
RANDOM_STATE         = 42
MIN_SAMPLES_PER_CLASS = 2         # guard against degenerate splits


# ── ActiveRetrainer ───────────────────────────────────────────────────────────

class ActiveRetrainer:
    """
    Monitors user corrections and retrains the expense classifier via
    active learning when enough corrections accumulate.

    Args:
        model_path:           Path to the ``.joblib`` model file.
        db_session:           Live SQLAlchemy ``Session`` connected to
                              ``expenses.db``.
        correction_threshold: Minimum number of new corrections before
                              retraining is triggered.
        csv_path:             Path to the original ``transactions.csv``
                              used as the base training corpus.
        log_path:             Path to the JSON file that records retrain events.
    """

    def __init__(
        self,
        model_path: str | Path,
        db_session: Session,
        correction_threshold: int = 20,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        log_path: str | Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.model_path  = Path(model_path)
        self.db          = db_session
        self.threshold   = correction_threshold
        self.csv_path    = Path(csv_path)
        self.log_path    = Path(log_path)

        log.info(
            "ActiveRetrainer initialised | model=%s  threshold=%d  csv=%s",
            self.model_path, self.threshold, self.csv_path,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_correction_count(self) -> int:
        """
        Return the total number of rows in ``ModelCorrection``.

        Returns:
            Integer row count.
        """
        count = self.db.query(ModelCorrection).count()
        log.debug("Correction count: %d", count)
        return count

    def check_and_retrain(self) -> bool:
        """
        Check whether the correction threshold has been reached and, if so,
        trigger a full retrain cycle.

        Returns:
            ``True`` if retraining was attempted (regardless of whether the
            new model was accepted), ``False`` if the threshold has not yet
            been reached.
        """
        count = self.get_correction_count()
        log.info("Checking threshold: %d corrections / %d required", count, self.threshold)

        if count < self.threshold:
            log.info("Threshold not met — skipping retrain.")
            return False

        log.info("Threshold met (%d >= %d) — starting retrain.", count, self.threshold)

        # ── Load base training data ───────────────────────────────────────
        orig_desc, orig_labels = self._load_original_data()
        if not orig_desc:
            log.error("No original training data found — aborting.")
            return True  # attempted but failed

        # ── Load corrections ──────────────────────────────────────────────
        corr_desc, corr_labels = self._load_corrections()
        if not corr_desc:
            log.warning("No usable corrections found in the database.")

        self.retrain(orig_desc, orig_labels, corr_desc, corr_labels)
        return True

    def retrain(
        self,
        original_descriptions: list[str],
        original_labels: list[str],
        correction_descriptions: list[str],
        correction_labels: list[str],
    ) -> None:
        """
        Combine original data with weighted corrections, retrain the pipeline,
        and save only if the new model improves on macro-F1.

        Corrections are duplicated ``CORRECTION_WEIGHT`` times so that
        recently corrected examples have a proportionally larger influence
        on the decision boundary.

        Args:
            original_descriptions:  Base training descriptions.
            original_labels:        Base training labels.
            correction_descriptions: Descriptions from user corrections.
            correction_labels:       Corrected labels.
        """
        log.info(
            "Retrain called | original=%d  corrections=%d (×%d weight)",
            len(original_descriptions), len(correction_descriptions), CORRECTION_WEIGHT,
        )

        # ── Combine with correction weighting ─────────────────────────────
        all_desc   = original_descriptions + correction_descriptions * CORRECTION_WEIGHT
        all_labels = original_labels       + correction_labels       * CORRECTION_WEIGHT
        log.info("Combined corpus size: %d samples", len(all_desc))

        # ── Evaluate current (old) model ──────────────────────────────────
        old_f1 = self._evaluate_current_model(all_desc, all_labels)

        # ── Train / test split ────────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                all_desc,
                all_labels,
                test_size=TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=all_labels,
            )
        except ValueError as exc:
            log.error("Train/test split failed (%s) — aborting retrain.", exc)
            return

        log.info("Split: train=%d  test=%d", len(X_train), len(X_test))

        # ── Train new classifier ──────────────────────────────────────────
        new_clf = ExpenseClassifier()
        log.info("Training new pipeline …")
        new_clf.train(X_train, y_train)
        log.info("Training complete.")

        # ── Evaluate new model ────────────────────────────────────────────
        metrics  = new_clf.evaluate(X_test, y_test, print_report=True)
        new_f1   = metrics["macro_f1"]
        log.info("New macro-F1: %.4f  |  Old macro-F1: %.4f", new_f1, old_f1)

        # ── Accept / reject ───────────────────────────────────────────────
        n_corrections = len(correction_descriptions)
        if new_f1 >= old_f1:
            self._save_model_atomic(new_clf)
            self.log_retrain_event(old_f1, new_f1, n_corrections)
            log.info(
                "✓ New model accepted (F1 %.4f → %.4f) and saved to %s.",
                old_f1, new_f1, self.model_path,
            )
        else:
            log.warning(
                "✗ New model rejected: F1 %.4f < old F1 %.4f. Keeping current model.",
                new_f1, old_f1,
            )
            self.log_retrain_event(old_f1, new_f1, n_corrections, accepted=False)

    def log_retrain_event(
        self,
        old_f1: float,
        new_f1: float,
        n_corrections: int,
        accepted: bool = True,
    ) -> None:
        """
        Append a retrain event record to the JSON log file.

        Each record contains timestamp, F1 scores, correction count, and
        whether the new model was accepted.

        Args:
            old_f1:        Macro-F1 of the model before retraining.
            new_f1:        Macro-F1 of the newly trained model.
            n_corrections: Number of corrections used in this cycle.
            accepted:      Whether the new model replaced the old one.
        """
        entry: dict[str, Any] = {
            "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
            "old_f1":       round(old_f1, 6),
            "new_f1":       round(new_f1, 6),
            "delta_f1":     round(new_f1 - old_f1, 6),
            "n_corrections": n_corrections,
            "accepted":     accepted,
            "model_path":   str(self.model_path),
        }

        # Load existing log or start fresh
        records: list[dict[str, Any]] = []
        if self.log_path.exists():
            try:
                records = json.loads(self.log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read existing log (%s) — starting fresh.", exc)

        records.append(entry)

        try:
            self.log_path.write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            log.info("Retrain event logged → %s", self.log_path)
        except OSError as exc:
            log.error("Failed to write retrain log: %s", exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_original_data(self) -> tuple[list[str], list[str]]:
        """
        Load descriptions and labels from ``transactions.csv``.

        Rows with missing fields or categories outside ``CATEGORIES`` are
        silently dropped.

        Returns:
            Tuple of ``(descriptions, labels)``.
        """
        if not self.csv_path.exists():
            log.error("CSV not found: %s", self.csv_path)
            return [], []

        df = pd.read_csv(self.csv_path)
        before = len(df)
        df = df.dropna(subset=["description", "category"])
        df = df[df["category"].isin(CATEGORIES)]
        dropped = before - len(df)
        if dropped:
            log.debug("Dropped %d rows from CSV (missing/ambiguous).", dropped)

        log.info("Loaded %d rows from %s.", len(df), self.csv_path)
        return df["description"].tolist(), df["category"].tolist()

    def _load_corrections(self) -> tuple[list[str], list[str]]:
        """
        Load corrections from the ``ModelCorrection`` table, joining to
        ``Transaction`` to retrieve the description text.

        Returns:
            Tuple of ``(descriptions, corrected_labels)``.
        """
        rows = (
            self.db.query(ModelCorrection, Transaction)
            .join(Transaction, ModelCorrection.transaction_id == Transaction.id)
            .all()
        )

        descriptions: list[str] = []
        labels: list[str]       = []
        skipped = 0

        for correction, transaction in rows:
            label = correction.corrected_label
            desc  = transaction.description

            if not desc or not label:
                skipped += 1
                continue
            if label not in CATEGORIES:
                log.debug("Skipping unknown label %r from correction id=%d", label, correction.id)
                skipped += 1
                continue

            descriptions.append(str(desc))
            labels.append(str(label))

        log.info(
            "Loaded %d corrections from DB (%d skipped).",
            len(descriptions), skipped,
        )
        return descriptions, labels

    def _evaluate_current_model(
        self, descriptions: list[str], labels: list[str]
    ) -> float:
        """
        Evaluate the on-disk model against a held-out split of the combined
        corpus to establish a baseline macro-F1.

        Returns:
            Macro-F1 score, or ``0.0`` if the model file does not exist or
            evaluation fails.
        """
        if not self.model_path.exists():
            log.warning("No existing model found at %s — baseline F1 set to 0.", self.model_path)
            return 0.0

        try:
            clf = ExpenseClassifier.load(self.model_path)
        except Exception as exc:
            log.error("Could not load existing model (%s) — baseline F1 set to 0.", exc)
            return 0.0

        try:
            _, X_eval, _, y_eval = train_test_split(
                descriptions,
                labels,
                test_size=TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=labels,
            )
        except ValueError as exc:
            log.warning("Could not split for baseline eval (%s) — returning 0.", exc)
            return 0.0

        results    = clf.predict_batch(X_eval)
        y_pred     = [r["category"] for r in results]
        macro_f1   = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        log.info("Current model baseline macro-F1: %.4f  (n=%d)", macro_f1, len(X_eval))
        return float(macro_f1)

    def _save_model_atomic(self, clf: ExpenseClassifier) -> None:
        """
        Save ``clf`` to a temporary file in the same directory as
        ``model_path``, then atomically replace the target.

        Using ``os.replace`` (POSIX rename) ensures a live API process
        never reads a partially written file.

        Args:
            clf: Fitted ``ExpenseClassifier`` to persist.
        """
        target_dir = self.model_path.parent
        fd, tmp_path = tempfile.mkstemp(
            suffix=".joblib.tmp", dir=target_dir
        )
        try:
            os.close(fd)
            clf.save(tmp_path)
            os.replace(tmp_path, self.model_path)
            log.info("Atomic save complete: %s", self.model_path)
        except Exception as exc:
            log.error("Atomic save failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ── __main__ simulation ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    from datetime import date

    from database import SessionLocal, Transaction, ModelCorrection, init_db

    log.info("=" * 60)
    log.info("retrainer.py — active learning simulation")
    log.info("=" * 60)

    # ── Ensure schema exists ──────────────────────────────────────────────
    init_db()

    db = SessionLocal()

    # ── Seed a dummy user + transactions for the simulation ───────────────
    from database import User
    from sqlalchemy import text

    # Clean up prior simulation data so this is idempotent
    db.execute(text("DELETE FROM model_corrections"))
    db.execute(text("DELETE FROM transactions"))
    db.execute(text("DELETE FROM users"))
    db.commit()

    dummy_user = User(
        username="sim_user",
        email="sim@example.com",
        hashed_password="hashed_pw_placeholder",
    )
    db.add(dummy_user)
    db.flush()

    # Seed 25 transactions that will receive corrections
    SAMPLE_CORRECTIONS = [
        ("MCDONALDS PAYMENT",        "Food",          "Food"),
        ("NETFLIX.COM",              "Entertainment", "Entertainment"),
        ("SHELL PETROL STN",         "Transport",     "Transport"),
        ("AMAZON MKTPLACE PMT",      "Shopping",      "Shopping"),
        ("UNIVERSITY TUITION FEE",   "Education",     "Education"),
        ("WATER UTILITY PMT",        "Utilities",     "Utilities"),
        ("CVS PHARMACY 4821",        "Shopping",      "Healthcare"),   # ← mislabelled
        ("UBER TRIP 9921",           "Food",          "Transport"),    # ← mislabelled
        ("SPOTIFY AB",               "Utilities",     "Entertainment"),# ← mislabelled
        ("DENTAL OFFICE 0042",       "Shopping",      "Healthcare"),   # ← mislabelled
        ("HOME DEPOT 1234",          "Shopping",      "Housing"),      # ← mislabelled
        ("COURSERA SUB 99201",       "Shopping",      "Education"),    # ← mislabelled
        ("STARBUCKS STORE 0099",     "Food",          "Food"),
        ("COMCAST 55810",            "Entertainment", "Utilities"),    # ← mislabelled
        ("LABCORP 778833",           "Shopping",      "Healthcare"),   # ← mislabelled
        ("TACO BELL 4491",           "Food",          "Food"),
        ("LYFT RIDE 20291",          "Shopping",      "Transport"),    # ← mislabelled
        ("PARKING METER 0088",       "Food",          "Transport"),    # ← mislabelled
        ("HBO MAX 310001",           "Utilities",     "Entertainment"),# ← mislabelled
        ("NATIONAL RAIL 880012",     "Shopping",      "Transport"),    # ← mislabelled
        ("PHYSICAL THERAPY 2211",    "Shopping",      "Healthcare"),   # ← mislabelled
        ("STUDENT LOAN PMT 44810",   "Shopping",      "Education"),    # ← mislabelled
        ("ELECTRIC BILL 99012",      "Shopping",      "Utilities"),    # ← mislabelled
        ("BURGER KING 0331",         "Food",          "Food"),
        ("CITY MEDICAL CTR 0091",    "Shopping",      "Healthcare"),   # ← mislabelled
    ]

    transactions_created: list[Transaction] = []
    for desc, predicted, _ in SAMPLE_CORRECTIONS:
        txn = Transaction(
            user_id            = dummy_user.id,
            date               = datetime.now(tz=timezone.utc),
            description        = desc,
            amount             = round(random.uniform(5, 200), 2),
            predicted_category = predicted,
            corrected_category = None,
            confidence_score   = round(random.uniform(0.5, 0.99), 4),
        )
        db.add(txn)
        transactions_created.append(txn)

    db.flush()

    # ── Simulate 25 user corrections ─────────────────────────────────────
    log.info("Inserting 25 simulated corrections …")
    for txn, (_, predicted, corrected) in zip(transactions_created, SAMPLE_CORRECTIONS):
        if predicted != corrected:
            log.info(
                "  Correction: '%s'  %s → %s",
                txn.description, predicted, corrected,
            )
        correction = ModelCorrection(
            transaction_id      = txn.id,
            original_prediction = predicted,
            corrected_label     = corrected,
        )
        db.add(correction)

    db.commit()
    log.info("25 corrections committed to DB.")

    # ── Run active retrainer ──────────────────────────────────────────────
    retrainer = ActiveRetrainer(
        model_path           = DEFAULT_MODEL_PATH,
        db_session           = db,
        correction_threshold = 20,
        csv_path             = DEFAULT_CSV_PATH,
        log_path             = DEFAULT_LOG_PATH,
    )

    log.info("Correction count: %d", retrainer.get_correction_count())
    triggered = retrainer.check_and_retrain()
    log.info("Retrain triggered: %s", triggered)

    # ── Print log summary ─────────────────────────────────────────────────
    if DEFAULT_LOG_PATH.exists():
        records = json.loads(DEFAULT_LOG_PATH.read_text())
        log.info("-" * 60)
        log.info("Retrain log (%s):", DEFAULT_LOG_PATH)
        for rec in records:
            log.info(
                "  [%s]  old_F1=%.4f  new_F1=%.4f  Δ=%+.4f  n_corr=%d  accepted=%s",
                rec["timestamp"][:19],
                rec["old_f1"], rec["new_f1"], rec["delta_f1"],
                rec["n_corrections"], rec["accepted"],
            )

    db.close()
    log.info("=" * 60)
    log.info("Simulation complete.")
    log.info("=" * 60)
