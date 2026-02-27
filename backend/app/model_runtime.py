from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "fraud_model.pt"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
META_PATH = ARTIFACT_DIR / "meta.joblib"

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "device_changed",
    "location_changed",
    "recent_txn_count_1h",
    "type_PAYMENT",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_CASH_IN",
    "type_DEBIT",
]


class FraudNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelRuntime:
    def __init__(self) -> None:
        self.ready = False
        self.model: FraudNet | None = None
        self.scaler = None

        if MODEL_PATH.exists() and SCALER_PATH.exists():
            payload = torch.load(MODEL_PATH, map_location="cpu")
            input_dim = payload["input_dim"]
            self.model = FraudNet(input_dim=input_dim)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
            self.scaler = joblib.load(SCALER_PATH)
            self.ready = True

    def _to_dataframe(self, txs: List[Dict[str, Any]]) -> pd.DataFrame:
        base = pd.DataFrame(txs)
        base["device_changed"] = base["device_changed"].astype(int)
        base["location_changed"] = base["location_changed"].astype(int)

        type_dummies = pd.get_dummies(base["type"], prefix="type")
        frame = pd.concat([base.drop(columns=["type", "language", "transaction_id"]), type_dummies], axis=1)

        for col in FEATURE_COLUMNS:
            if col not in frame.columns:
                frame[col] = 0

        return frame[FEATURE_COLUMNS].astype(float)

    def _heuristic_prob(self, tx: Dict[str, Any]) -> float:
        score = 0.05
        if tx["type"] in {"TRANSFER", "CASH_OUT"}:
            score += 0.15
        if tx["amount"] > 200000:
            score += 0.35
        if tx["recent_txn_count_1h"] >= 6:
            score += 0.20
        if tx["device_changed"]:
            score += 0.10
        if tx["location_changed"]:
            score += 0.10
        if tx["oldbalanceOrg"] > 0 and tx["newbalanceOrig"] <= 0:
            score += 0.10
        return float(np.clip(score, 0.0, 0.99))

    def predict_probabilities(self, txs: List[Dict[str, Any]]) -> List[float]:
        if not txs:
            return []

        if self.ready and self.model is not None and self.scaler is not None:
            features = self._to_dataframe(txs)
            scaled = self.scaler.transform(features.values)
            tensor_x = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(tensor_x).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
            return [float(p) for p in probs]

        return [self._heuristic_prob(tx) for tx in txs]
