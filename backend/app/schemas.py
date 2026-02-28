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
    unknown_qr_code: bool = False
    asked_scan_to_receive_money: bool = False
    pin_or_otp_prompt_after_qr: bool = False
    collect_request_received: bool = False
    otp_shared: bool = False
    upi_pin_shared: bool = False
    phishing_link_clicked: bool = False
    remote_app_installed: bool = False
    screen_share_active: bool = False
    fake_payment_screenshot: bool = False
    merchant_name_mismatch: bool = False
    urgency_pressure: bool = False
    unknown_caller_request: bool = False
    suspicious_app_clone: bool = False


class PredictionResult(BaseModel):
    transaction_id: str
    risk_score: float
    fraud_probability: float
    blocked: bool
    verdict: Literal["SAFE", "RISKY", "BLOCKED"]
    tip: str
    language: LanguageCode
    fraud_categories: List[str]
    advice: List[str]


class BatchPredictRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictResponse(BaseModel):
    total: int
    blocked: int
    risky: int
    safe: int
    results: List[PredictionResult]
