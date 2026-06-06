# Spendly — AI Expense Tracker

> A full-stack AI web application that automatically categorizes bank statement transactions using a machine learning text classifier, with active learning that improves the model over time.

![Tech Stack](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Overview

Spendly lets users upload bank statement CSV files and automatically categorizes every transaction — Food, Transport, Housing, Entertainment, and more — using a TF-IDF + Logistic Regression classifier trained on transaction descriptions.

What sets it apart is the **active learning loop**: when users correct a wrong prediction, those corrections accumulate in the database and periodically retrain the model, so it gets smarter over time. **SHAP explainability** shows users exactly which keywords drove each prediction.

---

## Features

- **Automatic categorization** — TF-IDF + Logistic Regression across 8 spending classes
- **Active learning** — model retrains itself as users correct wrong predictions, with F1 guard to prevent regressions
- **SHAP explainability** — per-prediction keyword attribution shown directly in the UI
- **Real CSV parsing** — handles messy, inconsistent bank statement formats
- **Interactive dashboard** — spending charts, monthly trends, and budget tracking via Chart.js
- **JWT authentication** — secure user accounts with protected endpoints
- **Anomaly detection** — flags unusual transactions more than 2 standard deviations above the category mean

---

## How It Works

The ML pipeline runs in three stages:

1. **Preprocessing** — Transaction descriptions are cleaned and normalized: merchant codes stripped, text uppercased, whitespace collapsed.
2. **Vectorization** — A TF-IDF vectorizer converts descriptions into numerical features using character n-grams (2–4 chars) and word n-grams (1–2 words), allowing partial matches like `MCDON` → McDonald's.
3. **Classification** — A Logistic Regression classifier assigns one of 8 spending categories with a confidence score.

When a user corrects a prediction, it's stored in the database. Once 20 corrections accumulate, the active retrainer combines original training data with corrections (weighted 3×) and retrains the full pipeline. The new model only replaces the old one if its F1 score improves — quality never regresses.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS · Chart.js |
| Backend | FastAPI · Python 3.11 |
| Database | SQLite · SQLAlchemy |
| ML Model | scikit-learn · TF-IDF · Logistic Regression |
| Explainability | SHAP |
| Containerization | Docker |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized run)

### Installation

```bash
git clone https://github.com/UsmanDanial04/spendly.git
cd spendly
pip install -r requirements.txt
```

### Train the Model

Generate the synthetic training dataset, then train the classifier:

```bash
python generate_dataset.py
python model_tfidf.py
```

### Run the App

Initialize the database and start the API server:

```bash
python database.py
uvicorn main:app --reload --port 8000
```

Open `index.html` in your browser or serve it with any static server. The frontend connects to `http://localhost:8000` by default.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key` | JWT signing secret — **change before any public deployment** |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `ENV` | `development` | Set to `production` for deployment |

---

## API Reference

### Authentication

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/auth/register` | Create a new user account | No |
| `POST` | `/auth/login` | Login and receive a JWT token | No |
| `GET` | `/health` | API health check | No |

### Transactions

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/upload` | Upload a bank statement CSV for bulk prediction | Yes |
| `GET` | `/transactions` | List all transactions (filter by category/month) | Yes |
| `POST` | `/predict` | Predict category for a single transaction | Yes |
| `PATCH` | `/correct/{id}` | Submit a category correction | Yes |
| `DELETE` | `/transactions/{id}` | Delete a transaction record | Yes |

### Insights & Analytics

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/insights/summary` | Monthly spending totals by category | Yes |
| `GET` | `/insights/trends` | Month-over-month spending for the last 6 months | Yes |
| `GET` | `/insights/anomalies` | Transactions flagged as statistical outliers | Yes |
| `GET` | `/insights/budget-status` | Budget usage per category with warning thresholds | Yes |

---

## ML Model Performance

Baseline model trained on 2,000 synthetic transactions, evaluated on a held-out 20% test split.

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Food | — | — | — | ~50 |
| Transport | — | — | — | ~50 |
| Housing | — | — | — | ~50 |
| Entertainment | — | — | — | ~50 |
| Healthcare | — | — | — | ~50 |
| Shopping | — | — | — | ~50 |
| Education | — | — | — | ~50 |
| Utilities | — | — | — | ~50 |
| **Overall Accuracy** | | | **—** | **400** |

> Run `python model_tfidf.py` to generate the full classification report and fill in these metrics.

---

## What I Learned

**ML in production** — Handling real-world bank statement noise required character n-gram feature engineering. A 2–4 char window captures truncated merchant names that word-level tokenizers miss entirely.

**Active learning design** — The retrainer's core challenge was preventing catastrophic forgetting while still shifting the model. Weighting corrections 3× provided a strong enough signal without discarding the original distribution. The F1 guard ensured quality never regressed during early-stage corrections.

**Explainability as UX** — SHAP is typically used for model debugging. Surfacing it directly to users — "classified as Food because: mcdonalds, burger" — reduced correction friction and increased trust. Transparency became a product feature, not just a diagnostic tool.

**Full-stack integration** — Wiring a real CSV parser to a live ML pipeline to a FastAPI backend to a vanilla JS frontend exposed integration points that tutorials abstract away: content-type headers, FormData handling, CORS configuration, and async vs. synchronous model inference tradeoffs.

---

## Future Improvements

- Swap TF-IDF for a fine-tuned DistilBERT model for semantically ambiguous descriptions
- Redis-backed job queue so retraining runs in a background worker, not on the request thread
- Multi-currency support with automatic exchange rate normalization
- Recurring transaction detection using time-series clustering to flag subscriptions and bills
- PDF/CSV export with embedded Chart.js visualizations

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Usman Danial**
- GitHub: [github.com/UsmanDanial04](https://github.com/UsmanDanial04)
- LinkedIn: [linkedin.com/in/usman-danial-b4568b289](https://www.linkedin.com/in/usman-danial-b4568b289)
