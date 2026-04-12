from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def analytics_db(tmp_path: Path) -> Path:
    """Minimal gaming_mental_health SQLite DB with 3 known columns."""
    db = tmp_path / "gaming_mental_health.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE gaming_mental_health (
            age INTEGER,
            gender TEXT,
            playtime_hours REAL
        )
        """
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def schema_description_db(tmp_path: Path) -> Path:
    """Minimal schema_metadata SQLite DB with table + 2 column descriptions."""
    db = tmp_path / "schema_metadata.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE table_descriptions (
            id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE column_descriptions (
            id TEXT PRIMARY KEY,
            table_description_id TEXT NOT NULL REFERENCES table_descriptions(id),
            column_name TEXT NOT NULL,
            domain TEXT,
            description TEXT NOT NULL,
            UNIQUE (table_description_id, column_name)
        )
        """
    )
    table_id = "test-table-id"
    conn.execute(
        "INSERT INTO table_descriptions VALUES (?, ?, ?)",
        (table_id, "gaming_mental_health", "Gaming and Mental Health Behavioral Dataset"),
    )
    conn.execute(
        "INSERT INTO column_descriptions VALUES (?, ?, ?, ?, ?)",
        ("col-age-id", table_id, "age", "Demographic", "Age of the participant"),
    )
    conn.execute(
        "INSERT INTO column_descriptions VALUES (?, ?, ?, ?, ?)",
        ("col-gender-id", table_id, "gender", "Demographic", "Gender category"),
    )
    conn.commit()
    conn.close()
    return db
