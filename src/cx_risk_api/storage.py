"""SQLite persistence for API score records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from cx_risk.config import OUTPUT_DIR
from cx_risk.utils import ensure_directory

DEFAULT_DB_PATH = OUTPUT_DIR / "scoring_api.db"


class ScoreStore:
    """Tiny SQLite store for scored-order audit records."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        ensure_directory(self.db_path.parent)
        self.initialize()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    order_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_band TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    reason_summary TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )

    def insert_score(self, response: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scores (
                    order_id,
                    model,
                    risk_score,
                    risk_band,
                    recommended_action,
                    reason_summary,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response["order_id"],
                    response["model"],
                    response["risk_score"],
                    response["risk_band"],
                    response["recommended_action"],
                    response["reason_summary"],
                    json.dumps(response),
                ),
            )

    def recent_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    scored_at,
                    order_id,
                    model,
                    risk_score,
                    risk_band,
                    recommended_action,
                    reason_summary,
                    response_json
                FROM scores
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["response"] = json.loads(record.pop("response_json"))
            records.append(record)
        return records

