from __future__ import annotations

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "upi_sentinel.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                risk_score REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                blocked INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                tip TEXT NOT NULL,
                language TEXT NOT NULL,
                fraud_categories_json TEXT NOT NULL,
                advice_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(full_name: str, email: str, password: str) -> dict[str, Any]:
    clean_name = full_name.strip()
    clean_email = email.strip().lower()
    if not clean_name:
        raise ValueError("Full name is required")
    if not clean_email:
        raise ValueError("Email is required")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    conn = _get_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,)).fetchone()
        if existing:
            raise ValueError("User already registered with this email")

        created_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO users (full_name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (clean_name, clean_email, _hash_password(password), created_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, full_name, email, created_at FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
    finally:
        conn.close()

    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def verify_user(email: str, password: str) -> dict[str, Any] | None:
    clean_email = email.strip().lower()
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, full_name, email, password_hash, created_at
            FROM users
            WHERE email = ?
            """,
            (clean_email,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    if row["password_hash"] != _hash_password(password):
        return None

    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def save_scored_transaction(payload: dict[str, Any], result: dict[str, Any]) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO scored_transactions (
                transaction_id,
                payload_json,
                risk_score,
                fraud_probability,
                blocked,
                verdict,
                tip,
                language,
                fraud_categories_json,
                advice_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["transaction_id"],
                json.dumps(payload, ensure_ascii=False),
                float(result["risk_score"]),
                float(result["fraud_probability"]),
                int(bool(result["blocked"])),
                result["verdict"],
                result["tip"],
                result["language"],
                json.dumps(result.get("fraud_categories", []), ensure_ascii=False),
                json.dumps(result.get("advice", []), ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_transactions(limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 500))
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                transaction_id,
                payload_json,
                risk_score,
                fraud_probability,
                blocked,
                verdict,
                tip,
                language,
                fraud_categories_json,
                advice_json,
                created_at
            FROM scored_transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "transaction_id": row["transaction_id"],
                "payload": json.loads(row["payload_json"]),
                "risk_score": row["risk_score"],
                "fraud_probability": row["fraud_probability"],
                "blocked": bool(row["blocked"]),
                "verdict": row["verdict"],
                "tip": row["tip"],
                "language": row["language"],
                "fraud_categories": json.loads(row["fraud_categories_json"]),
                "advice": json.loads(row["advice_json"]),
                "created_at": row["created_at"],
            }
        )
    return out
