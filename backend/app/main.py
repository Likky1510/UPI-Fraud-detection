from __future__ import annotations

from typing import Dict
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .i18n import get_tip
from .model_runtime import ModelRuntime
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
    TransactionRequest,
)

app = FastAPI(title="UPI Shield API", version="1.0.0")
model_runtime = ModelRuntime()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rule_risk(tx: Dict) -> float:
    score = 0.0

    if tx["type"] in {"TRANSFER", "CASH_OUT"}:
        score += 18
    if tx["amount"] >= 50000:
        score += 18
    if tx["amount"] >= 200000:
        score += 12
    if tx["recent_txn_count_1h"] >= 5:
        score += 20
    if tx["device_changed"]:
        score += 15
    if tx["location_changed"]:
        score += 12

    if tx["oldbalanceOrg"] > 0:
        balance_drain_ratio = (tx["oldbalanceOrg"] - tx["newbalanceOrig"]) / max(tx["oldbalanceOrg"], 1)
        if balance_drain_ratio > 0.95:
            score += 10

    return min(score, 100.0)


def score_transaction(tx: TransactionRequest, fraud_prob: float) -> PredictionResult:
    tx_data = tx.model_dump()
    model_score = fraud_prob * 100.0
    policy_score = rule_risk(tx_data)
    final_score = max(0.0, min(100.0, 0.7 * model_score + 0.3 * policy_score))

    if final_score >= 80:
        verdict = "BLOCKED"
        blocked = True
    elif final_score >= 55:
        verdict = "RISKY"
        blocked = False
    else:
        verdict = "SAFE"
        blocked = False

    return PredictionResult(
        transaction_id=tx.transaction_id,
        risk_score=round(final_score, 2),
        fraud_probability=round(fraud_prob, 4),
        blocked=blocked,
        verdict=verdict,
        tip=get_tip(tx.language, verdict),
        language=tx.language,
    )


def build_simulated_tx(index: int, language: str) -> dict:
    tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount = round(random.uniform(100, 300000), 2)
    old_org = round(random.uniform(amount, amount + 300000), 2)
    new_org = max(0.0, round(old_org - amount + random.uniform(-200, 300), 2))

    return {
        "transaction_id": f"SIM-{index + 1}",
        "step": random.randint(1, 744),
        "type": random.choice(tx_types),
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": round(random.uniform(0, 400000), 2),
        "newbalanceDest": round(random.uniform(0, 500000), 2),
        "language": language if language in {"en", "hi", "te"} else "en",
        "device_changed": random.random() < 0.2,
        "location_changed": random.random() < 0.18,
        "recent_txn_count_1h": random.randint(0, 8),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": model_runtime.ready,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(tx: TransactionRequest) -> PredictionResult:
    prob = model_runtime.predict_probabilities([tx.model_dump()])[0]
    return score_transaction(tx, prob)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    txs = payload.transactions
    probs = model_runtime.predict_probabilities([tx.model_dump() for tx in txs])
    results = [score_transaction(tx, prob) for tx, prob in zip(txs, probs)]

    blocked = sum(1 for r in results if r.verdict == "BLOCKED")
    risky = sum(1 for r in results if r.verdict == "RISKY")
    safe = sum(1 for r in results if r.verdict == "SAFE")

    return BatchPredictResponse(
        total=len(results),
        blocked=blocked,
        risky=risky,
        safe=safe,
        results=results,
    )


@app.get("/sample-transactions")
def sample_transactions(count: int = 1200, language: str = "en") -> dict:
    count = max(1, min(count, 5000))
    rows = [build_simulated_tx(i, language) for i in range(count)]
    return {"transactions": rows}


@app.get("/simulate")
def simulate(count: int = 1200, language: str = "en", chunk_size: int = 5000, preview: int = 25) -> dict:
    count = max(1, min(count, 250000))
    chunk_size = max(500, min(chunk_size, 20000))
    preview = max(1, min(preview, 100))

    blocked = 0
    risky = 0
    safe = 0
    preview_rows = []
    processed = 0

    while processed < count:
        current_size = min(chunk_size, count - processed)
        tx_dicts = [build_simulated_tx(processed + i, language) for i in range(current_size)]
        tx_models = [TransactionRequest(**tx) for tx in tx_dicts]
        probs = model_runtime.predict_probabilities(tx_dicts)

        for tx_model, tx_data, prob in zip(tx_models, tx_dicts, probs):
            result = score_transaction(tx_model, prob)
            if result.verdict == "BLOCKED":
                blocked += 1
            elif result.verdict == "RISKY":
                risky += 1
            else:
                safe += 1

            if len(preview_rows) < preview:
                preview_rows.append(
                    {
                        "transaction_id": result.transaction_id,
                        "type": tx_data["type"],
                        "amount": tx_data["amount"],
                        "verdict": result.verdict,
                        "risk_score": result.risk_score,
                        "tip": result.tip,
                    }
                )

        processed += current_size

    return {
        "total": count,
        "blocked": blocked,
        "risky": risky,
        "safe": safe,
        "preview_rows": preview_rows,
    }
