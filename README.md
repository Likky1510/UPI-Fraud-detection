# UPI-Fraud-Detection-Awareness-System

# UPI Shield - Fraud Detection Web Interface

This project now includes:
- Deep learning training pipeline for PaySim synthetic financial data
- FastAPI backend for real-time fraud scoring and blocking
- Multilingual awareness tips in English, Hindi, Telugu
- Frontend dashboard integrated with backend APIs

## 1) Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Train model (PaySim)

Download the PaySim CSV and run:

```bash
python train\train_model.py --csv "D:\path\to\PS_20174392719_1491204439457_log.csv" --epochs 10
```

Artifacts are saved in `backend/artifacts/`:
- `fraud_model.pt`
- `scaler.joblib`
- `meta.joblib`

## 3) Start API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4) Start frontend

Open `UPI Sheild/dashboard.html` in a local server (recommended) so API calls work cleanly.

## API endpoints

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /sample-transactions?count=1200&language=en`

## Risk policy

- `score >= 80` => `BLOCKED`
- `55 <= score < 80` => `RISKY`
- `< 55` => `SAFE`

Combined score = `70% model probability + 30% rule-based signals`
