from __future__ import annotations

from typing import Dict
import random

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .i18n import get_tip
from .db import create_user, fetch_recent_transactions, init_db, save_scored_transaction, verify_user
from .model_runtime import ModelRuntime
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
    TransactionRequest,
)
from .tts_service import CloudTTSService

app = FastAPI(title="UPI Sentinel API", version="1.0.0")
model_runtime = ModelRuntime()
tts_service = CloudTTSService()
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRAUD_LABELS = {
    "qr_scam": {"en": "QR Code Scam", "hi": "QR कोड धोखाधड़ी", "te": "QR కోడ్ మోసం"},
    "phishing": {"en": "Phishing Link Fraud", "hi": "फिशिंग लिंक धोखाधड़ी", "te": "ఫిషింగ్ లింక్ మోసం"},
    "otp_fraud": {"en": "OTP Fraud", "hi": "OTP धोखाधड़ी", "te": "OTP మోసం"},
    "upi_pin_fraud": {"en": "UPI PIN Fraud", "hi": "UPI PIN धोखाधड़ी", "te": "UPI PIN మోసం"},
    "fake_payment_proof": {
        "en": "Fake UPI Payment Screenshot",
        "hi": "फर्जी UPI भुगतान स्क्रीनशॉट",
        "te": "నకిలీ UPI చెల్లింపు స్క్రీన్‌షాట్",
    },
    "collect_request_scam": {"en": "Collect Request Scam", "hi": "कलेक्ट रिक्वेस्ट धोखाधड़ी", "te": "కలెక్ట్ రిక్వెస్ట్ మోసం"},
    "remote_access_scam": {"en": "Remote Access App Scam", "hi": "रिमोट ऐप धोखाधड़ी", "te": "రిమోట్ యాప్ మోసం"},
    "screen_share_scam": {"en": "Screen Share Scam", "hi": "स्क्रीन शेयर धोखाधड़ी", "te": "స్క్రీన్ షేర్ మోసం"},
    "social_engineering": {"en": "Social Engineering Pressure", "hi": "सोशल इंजीनियरिंग दबाव", "te": "సోషల్ ఇంజనీరింగ్ ఒత్తిడి"},
    "caller_impersonation": {"en": "Impersonation Call Scam", "hi": "फर्जी कॉल धोखाधड़ी", "te": "నకిలీ కాల్ మోసం"},
    "merchant_mismatch": {"en": "Merchant Name Mismatch", "hi": "मर्चेंट नाम असंगति", "te": "మర్చెంట్ పేరు అసమానం"},
    "fake_upi_app": {"en": "Fake UPI App Clone", "hi": "फर्जी UPI ऐप क्लोन", "te": "నకిలీ UPI యాప్ క్లోన్"},
    "qr_receive_trick": {"en": "Scan-to-Receive QR Trick", "hi": "स्कैन करके पैसा पाने वाला झांसा", "te": "స్కాన్ చేసి డబ్బు వస్తుందన్న మోసం"},
    "qr_pin_prompt_trick": {"en": "PIN/OTP Prompt After QR", "hi": "QR के बाद PIN/OTP मांगना", "te": "QR తర్వాత PIN/OTP అడగడం"},
}

FRAUD_ADVICE = {
    "en": {
        "qr_scam": "Never scan unknown QR codes for receiving money.",
        "phishing": "Do not open unknown payment links from SMS/WhatsApp/email.",
        "otp_fraud": "Never share OTP with anyone, including bank support.",
        "upi_pin_fraud": "UPI PIN is only for sending money. Never disclose it.",
        "fake_payment_proof": "Verify credit in your app/bank SMS, not screenshots.",
        "collect_request_scam": "Decline suspicious collect requests from unknown IDs.",
        "remote_access_scam": "Do not install remote apps (AnyDesk/TeamViewer) for support calls.",
        "screen_share_scam": "Never share your screen while doing UPI transactions.",
        "social_engineering": "Scammers create urgency. Pause and verify first.",
        "caller_impersonation": "Disconnect and call official bank helpline directly.",
        "merchant_mismatch": "Confirm payee name before entering UPI PIN.",
        "fake_upi_app": "Install UPI apps only from official app stores.",
        "qr_receive_trick": "To receive money, no QR scan or UPI PIN is required.",
        "qr_pin_prompt_trick": "If QR flow asks for OTP/PIN to receive money, cancel immediately.",
    },
    "hi": {
        "qr_scam": "पैसा प्राप्त करने के लिए अज्ञात QR कोड स्कैन न करें।",
        "phishing": "SMS/WhatsApp/ईमेल के अज्ञात पेमेंट लिंक न खोलें।",
        "otp_fraud": "OTP किसी से साझा न करें, बैंक सपोर्ट से भी नहीं।",
        "upi_pin_fraud": "UPI PIN केवल भुगतान भेजने के लिए है, साझा न करें।",
        "fake_payment_proof": "स्क्रीनशॉट नहीं, बैंक ऐप/SMS में क्रेडिट जांचें।",
        "collect_request_scam": "अज्ञात UPI ID की कलेक्ट रिक्वेस्ट अस्वीकार करें।",
        "remote_access_scam": "सपोर्ट कॉल पर रिमोट ऐप इंस्टॉल न करें।",
        "screen_share_scam": "UPI लेनदेन के समय स्क्रीन शेयर न करें।",
        "social_engineering": "जल्दी कराने का दबाव धोखाधड़ी का संकेत है।",
        "caller_impersonation": "कॉल काटें और आधिकारिक बैंक हेल्पलाइन पर कॉल करें।",
        "merchant_mismatch": "UPI PIN डालने से पहले प्राप्तकर्ता नाम जांचें।",
        "fake_upi_app": "केवल आधिकारिक स्टोर से UPI ऐप इंस्टॉल करें।",
        "qr_receive_trick": "पैसा प्राप्त करने के लिए QR स्कैन या UPI PIN की जरूरत नहीं होती।",
        "qr_pin_prompt_trick": "यदि QR के बाद OTP/PIN मांगा जाए, तुरंत रद्द करें।",
    },
    "te": {
        "qr_scam": "డబ్బు స్వీకరించడానికి తెలియని QR కోడ్ స్కాన్ చేయవద్దు.",
        "phishing": "తెలియని SMS/WhatsApp/ఇమెయిల్ పేమెంట్ లింక్‌లు ఓపెన్ చేయవద్దు.",
        "otp_fraud": "OTP ఎవరితోనూ పంచుకోవద్దు, బ్యాంక్ సపోర్ట్‌తో కూడా కాదు.",
        "upi_pin_fraud": "UPI PIN డబ్బు పంపడానికి మాత్రమే. ఎప్పుడూ చెప్పవద్దు.",
        "fake_payment_proof": "స్క్రీన్‌షాట్ కాదండి, యాప్/SMS లో క్రెడిట్ వచ్చిందో చూడండి.",
        "collect_request_scam": "తెలియని UPI IDల నుండి కలెక్ట్ రిక్వెస్ట్‌ను తిరస్కరించండి.",
        "remote_access_scam": "సపోర్ట్ పేరుతో రిమోట్ యాప్‌లు ఇన్‌స్టాల్ చేయవద్దు.",
        "screen_share_scam": "UPI లావాదేవీ సమయంలో స్క్రీన్ షేర్ చేయవద్దు.",
        "social_engineering": "అత్యవసర ఒత్తిడి మోసానికి సంకేతం కావచ్చు.",
        "caller_impersonation": "కాల్ నిలిపి అధికారిక బ్యాంక్ నంబర్‌కు మళ్లీ కాల్ చేయండి.",
        "merchant_mismatch": "UPI PIN ఇవ్వడానికి ముందు పేరు సరైనదో చూడండి.",
        "fake_upi_app": "UPI యాప్‌లను అధికారిక స్టోర్ నుంచే ఇన్‌స్టాల్ చేయండి.",
        "qr_receive_trick": "డబ్బు స్వీకరించడానికి QR స్కాన్ లేదా UPI PIN అవసరం లేదు.",
        "qr_pin_prompt_trick": "QR తర్వాత OTP/PIN అడిగితే వెంటనే రద్దు చేయండి.",
    },
}


def detect_fraud_context(tx: Dict, language: str) -> tuple[float, list[str], list[str], list[str]]:
    points = 0.0
    keys: list[str] = []

    checks = [
        ("unknown_qr_code", "qr_scam", 22),
        ("asked_scan_to_receive_money", "qr_receive_trick", 28),
        ("pin_or_otp_prompt_after_qr", "qr_pin_prompt_trick", 35),
        ("phishing_link_clicked", "phishing", 25),
        ("otp_shared", "otp_fraud", 35),
        ("upi_pin_shared", "upi_pin_fraud", 45),
        ("fake_payment_screenshot", "fake_payment_proof", 24),
        ("collect_request_received", "collect_request_scam", 18),
        ("remote_app_installed", "remote_access_scam", 28),
        ("screen_share_active", "screen_share_scam", 24),
        ("urgency_pressure", "social_engineering", 14),
        ("unknown_caller_request", "caller_impersonation", 18),
        ("merchant_name_mismatch", "merchant_mismatch", 20),
        ("suspicious_app_clone", "fake_upi_app", 26),
    ]

    for field, key, weight in checks:
        if tx.get(field, False):
            points += weight
            keys.append(key)

    lang = language if language in {"en", "hi", "te"} else "en"
    categories = [FRAUD_LABELS[k][lang] for k in keys]
    advice = [FRAUD_ADVICE[lang][k] for k in keys][:5]
    return min(points, 100.0), keys, categories, advice


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

    # Consistency check: if destination balances are known, credit change should broadly match amount.
    amount = max(float(tx.get("amount", 0.0)), 0.0)
    old_dest = max(float(tx.get("oldbalanceDest", 0.0)), 0.0)
    new_dest = max(float(tx.get("newbalanceDest", 0.0)), 0.0)
    if amount > 0 and (old_dest > 0 or new_dest > 0):
        credited = max(0.0, new_dest - old_dest)
        if credited < amount * 0.5:
            score += 22
        elif abs(credited - amount) > amount * 0.4:
            score += 10

    return min(score, 100.0)


def score_transaction(tx: TransactionRequest, fraud_prob: float) -> PredictionResult:
    tx_data = tx.model_dump()
    amount = max(float(tx.amount), 0.0)
    model_score = fraud_prob * 100.0
    policy_score = rule_risk(tx_data)
    context_score, context_keys, categories, advice = detect_fraud_context(tx_data, tx.language)
    final_score = max(0.0, min(100.0, 0.55 * model_score + 0.25 * policy_score + 0.20 * context_score))

    # Hard safety overrides for known high-confidence fraud patterns.
    if "upi_pin_fraud" in context_keys:
        if amount >= 1000:
            final_score = max(final_score, 95.0)
        elif amount > 0:
            final_score = max(final_score, 82.0)
        else:
            final_score = max(final_score, 60.0)
    if "otp_fraud" in context_keys:
        if amount > 0:
            final_score = max(final_score, 72.0)
        else:
            final_score = max(final_score, 52.0)
        if amount >= 50000 or (amount > 0 and tx.type in {"DEBIT", "TRANSFER", "CASH_OUT"}):
            final_score = max(final_score, 88.0)
    if amount > 0 and "otp_fraud" in context_keys and "caller_impersonation" in context_keys:
        final_score = max(final_score, 92.0)
    if amount > 0 and "remote_access_scam" in context_keys and "screen_share_scam" in context_keys:
        final_score = max(final_score, 92.0)
    if "phishing" in context_keys and amount >= 50000:
        final_score = max(final_score, 82.0)
    if amount > 0 and len(context_keys) >= 3:
        final_score = max(final_score, 75.0)
    if "qr_receive_trick" in context_keys and amount > 0:
        final_score = max(final_score, 74.0)
    if "qr_pin_prompt_trick" in context_keys:
        final_score = max(final_score, 85.0 if amount > 0 else 70.0)
    if "unknown_qr_code" in tx_data and tx_data.get("unknown_qr_code", False) and amount > 0:
        final_score = max(final_score, 62.0)

    # Fake payment screenshot fraud should be evidence-driven:
    # - if receiver credit clearly matches amount, this can be safe;
    # - if mismatch/no credit, escalate to risky/blocked.
    if "fake_payment_proof" in context_keys and amount > 0:
        old_dest = max(0.0, float(tx_data.get("oldbalanceDest", 0.0)))
        new_dest = max(0.0, float(tx_data.get("newbalanceDest", 0.0)))
        credited = max(0.0, new_dest - old_dest)
        has_dest_info = old_dest > 0 or new_dest > 0

        if has_dest_info and abs(credited - amount) <= amount * 0.2:
            # Credit matches amount closely -> likely a real payment confirmation.
            if "merchant_mismatch" not in context_keys and len(context_keys) <= 1:
                final_score = min(final_score, 42.0)
            else:
                final_score = max(final_score, 58.0)
        elif not has_dest_info:
            # No destination balance evidence provided; keep suspicious but not auto-blocked.
            final_score = max(final_score, 66.0)
        elif credited < amount * 0.5:
            final_score = max(final_score, 90.0)
        elif abs(credited - amount) > amount * 0.4:
            final_score = max(final_score, 84.0)
        else:
            final_score = max(final_score, 72.0)

    # Guardrail: zero-amount transactions should not be blocked by default.
    if amount <= 0:
        if "upi_pin_fraud" in context_keys:
            final_score = min(final_score, 79.0)
        elif "otp_fraud" in context_keys:
            final_score = min(final_score, 69.0)
        else:
            final_score = min(final_score, 35.0)

    # Fake app clone policy requested by user:
    # if app asks OTP/PIN -> blocked, else normal baseline.
    if "fake_upi_app" in context_keys:
        if "otp_fraud" in context_keys or "upi_pin_fraud" in context_keys:
            final_score = 96.0
        else:
            final_score = min(final_score, 35.0)

    # User policy override: Unknown Caller Pressure Scam is always blocked.
    if "caller_impersonation" in context_keys and "social_engineering" in context_keys:
        final_score = 96.0

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
        fraud_categories=categories,
        advice=advice,
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
        "unknown_qr_code": random.random() < 0.08,
        "asked_scan_to_receive_money": random.random() < 0.06,
        "pin_or_otp_prompt_after_qr": random.random() < 0.03,
        "collect_request_received": random.random() < 0.12,
        "otp_shared": random.random() < 0.03,
        "upi_pin_shared": random.random() < 0.02,
        "phishing_link_clicked": random.random() < 0.06,
        "remote_app_installed": random.random() < 0.03,
        "screen_share_active": random.random() < 0.03,
        "fake_payment_screenshot": random.random() < 0.09,
        "merchant_name_mismatch": random.random() < 0.08,
        "urgency_pressure": random.random() < 0.10,
        "unknown_caller_request": random.random() < 0.10,
        "suspicious_app_clone": random.random() < 0.03,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": model_runtime.ready,
    }


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    language: str = Field("en", pattern="^(en|hi|te)$")


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)


@app.post("/predict", response_model=PredictionResult)
def predict(tx: TransactionRequest) -> PredictionResult:
    prob = model_runtime.predict_probabilities([tx.model_dump()])[0]
    result = score_transaction(tx, prob)
    save_scored_transaction(tx.model_dump(), result.model_dump())
    return result


@app.post("/auth/register")
def auth_register(payload: RegisterRequest) -> dict:
    try:
        user = create_user(payload.full_name, payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Registration successful", "user": user}


@app.post("/auth/login")
def auth_login(payload: LoginRequest) -> dict:
    user = verify_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user": user}


@app.post("/tts")
def tts(payload: TTSRequest) -> Response:
    try:
        audio_bytes = tts_service.synthesize_mp3(payload.text, payload.language)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Cloud TTS unavailable. Verify Google credentials and package installation.",
        ) from exc

    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    txs = payload.transactions
    probs = model_runtime.predict_probabilities([tx.model_dump() for tx in txs])
    results = [score_transaction(tx, prob) for tx, prob in zip(txs, probs)]
    for tx, result in zip(txs, results):
        save_scored_transaction(tx.model_dump(), result.model_dump())

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


@app.get("/transactions/recent")
def transactions_recent(limit: int = 50) -> dict:
    return {"items": fetch_recent_transactions(limit)}


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
                        "fraud_categories": result.fraud_categories,
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

