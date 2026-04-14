"""Tests for SQLiteExecutor persistent connection behaviour (Phase 2).

WHY: verifies the connection is opened once at init, reused across run() calls,
enforces read-only access at the OS level, and cleans up correctly via close()
and context manager.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.pipeline import SQLiteExecutor
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


def test_connection_opened_at_init(analytics_db):
    executor = SQLiteExecutor(analytics_db)
    assert executor._conn is not None
    executor.close()


def test_run_reuses_connection(analytics_db_with_data):
    executor = SQLiteExecutor(analytics_db_with_data)
    conn_id = id(executor._conn)
    executor.run("SELECT age FROM gaming_mental_health")
    executor.run("SELECT age FROM gaming_mental_health")
    assert id(executor._conn) == conn_id
    executor.close()


def test_close_then_run_returns_error(analytics_db):
    # WHY: run() catches all exceptions and returns SQLExecutionOutput.error —
    # using a closed connection must surface as an error output, not silently
    # return empty rows
    executor = SQLiteExecutor(analytics_db)
    executor.close()
    out = executor.run("SELECT age FROM gaming_mental_health")
    assert out.error is not None


def test_nonexistent_db_raises_at_init():
    # WHY: mode=ro fails immediately at construction so the error surfaces at
    # startup rather than being deferred to the first run() call
    with pytest.raises(Exception):
        SQLiteExecutor(Path("/tmp/does_not_exist_no_such_file_xyz.sqlite"))


# ---------------------------------------------------------------------------
# Read-only enforcement
# ---------------------------------------------------------------------------


def test_read_only_enforced(analytics_db_with_data):
    executor = SQLiteExecutor(analytics_db_with_data)
    with pytest.raises(sqlite3.OperationalError):
        executor._conn.execute("INSERT INTO gaming_mental_health VALUES (99, 'X', 1.0)")
    executor.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_connection(analytics_db_with_data):
    with SQLiteExecutor(analytics_db_with_data) as executor:
        result = executor.run("SELECT age FROM gaming_mental_health")
        assert result.row_count > 0
    # WHY: after __exit__ the connection must be closed; any operation must raise
    with pytest.raises(Exception):
        executor._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Existing row-truncation logic preserved through persistent connection
# ---------------------------------------------------------------------------


def test_row_truncation_flagged(analytics_db_101_rows):
    with SQLiteExecutor(analytics_db_101_rows) as executor:
        out = executor.run("SELECT * FROM gaming_mental_health")
    assert out.rows_truncated is True
    assert len(out.rows) == 100


def test_exactly_100_rows_not_truncated(analytics_db_100_rows):
    with SQLiteExecutor(analytics_db_100_rows) as executor:
        out = executor.run("SELECT * FROM gaming_mental_health")
    assert out.rows_truncated is False
    assert len(out.rows) == 100
