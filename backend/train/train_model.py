from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
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


def load_and_prepare(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PaySim columns: {sorted(missing)}")

    x = df[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
    ].copy()
    y = df["isFraud"].astype(np.float32).values

    x = pd.concat([x.drop(columns=["type"]), pd.get_dummies(x["type"], prefix="type")], axis=1)

    for col in FEATURE_COLUMNS:
        if col not in x.columns:
            x[col] = 0

    return x[FEATURE_COLUMNS].astype(np.float32).values, y


def train(csv_path: Path, artifact_dir: Path, epochs: int, batch_size: int, learning_rate: float) -> None:
    x, y = load_and_prepare(csv_path)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    train_x = torch.tensor(x_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    val_x = torch.tensor(x_val_scaled, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = FraudNet(input_dim=train_x.shape[1])

    positives = max(float((y_train == 1).sum()), 1.0)
    negatives = max(float((y_train == 0).sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % 2 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x)
                val_probs = torch.sigmoid(val_logits).view(-1).numpy()
                val_pred = (val_probs >= 0.5).astype(int)
                auc = roc_auc_score(y_val, val_probs)
                avg_loss = epoch_loss / max(len(loader), 1)
                print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_auc={auc:.4f}")
                print(classification_report(y_val, val_pred, digits=4))

    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_x.shape[1],
            "feature_columns": FEATURE_COLUMNS,
        },
        artifact_dir / "fraud_model.pt",
    )

    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump({"feature_columns": FEATURE_COLUMNS}, artifact_dir / "meta.joblib")

    print(f"Saved artifacts to: {artifact_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UPI fraud model using PaySim data")
    parser.add_argument("--csv", required=True, help="Path to PaySim CSV file")
    parser.add_argument("--artifact-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        csv_path=Path(args.csv),
        artifact_dir=Path(args.artifact_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
