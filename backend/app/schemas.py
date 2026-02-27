from pydantic import BaseModel, Field
from typing import Literal, List

LanguageCode = Literal["en", "hi", "te"]


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=2, max_length=64)
    step: int = Field(..., ge=1, le=744)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    language: LanguageCode = "en"
    device_changed: bool = False
    location_changed: bool = False
    recent_txn_count_1h: int = Field(0, ge=0, le=1000)


class PredictionResult(BaseModel):
    transaction_id: str
    risk_score: float
    fraud_probability: float
    blocked: bool
    verdict: Literal["SAFE", "RISKY", "BLOCKED"]
    tip: str
    language: LanguageCode


class BatchPredictRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictResponse(BaseModel):
    total: int
    blocked: int
    risky: int
    safe: int
    results: List[PredictionResult]
