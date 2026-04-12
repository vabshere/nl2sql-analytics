from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import SQLValidator

# Minimal schema context mirroring what load_schema_context() will return
_SCHEMA_CTX: dict = {
    "ddl": "CREATE TABLE gaming_mental_health (age INTEGER, gender TEXT, playtime_hours REAL)",
    "tables": {"gaming_mental_health"},
    "columns": {"gaming_mental_health": {"age", "gender", "playtime_hours"}},
}


# ---------------------------------------------------------------------------
# Layer 1 — statement-type + syntax (no schema_context)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [None, ""])
def test_null_and_empty_rejected(sql):
    result = SQLValidator.validate(sql)
    assert not result.is_valid
    assert result.error


@pytest.mark.parametrize("sql", [
    "DELETE FROM gaming_mental_health",
    "INSERT INTO gaming_mental_health VALUES(1, 'x', 2.0)",
    "UPDATE gaming_mental_health SET age=1",
    "DROP TABLE gaming_mental_health",
    "PRAGMA table_info(gaming_mental_health)",
])
def test_dml_ddl_rejected(sql):
    result = SQLValidator.validate(sql)
    assert not result.is_valid
    assert result.error


def test_multi_statement_rejected():
    result = SQLValidator.validate("SELECT 1; DELETE FROM gaming_mental_health")
    assert not result.is_valid
    assert result.error


@pytest.mark.parametrize("sql", [
    "select * from gaming_mental_health",
    "-- comment\nSELECT * FROM gaming_mental_health",
    "WITH x AS (SELECT age FROM gaming_mental_health) SELECT * FROM x",
])
def test_select_variants_accepted(sql):
    result = SQLValidator.validate(sql)
    assert result.is_valid
    assert result.validated_sql == sql
    assert result.error is None


# ---------------------------------------------------------------------------
# Layer 2 — schema + semantic validation (with schema_context)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT * FROM nonexistent_table",
])
def test_no_table_and_unknown_table_rejected(sql):
    result = SQLValidator.validate(sql, schema_context=_SCHEMA_CTX)
    assert not result.is_valid
    assert result.error


@pytest.mark.parametrize("sql", [
    "SELECT age FROM gaming_mental_health WHERE 1=1",
    "SELECT age FROM gaming_mental_health WHERE TRUE",
    "SELECT age FROM gaming_mental_health WHERE age > 5 OR 1=1",
])
def test_always_true_where_rejected(sql):
    result = SQLValidator.validate(sql, schema_context=_SCHEMA_CTX)
    assert not result.is_valid
    assert result.error


@pytest.mark.parametrize("sql", [
    "SELECT age FROM gaming_mental_health",
    "SELECT age FROM gaming_mental_health WHERE age > 5",
    "SELECT age, gender FROM gaming_mental_health WHERE playtime_hours > 10 ORDER BY age",
])
def test_valid_queries_pass(sql):
    result = SQLValidator.validate(sql, schema_context=_SCHEMA_CTX)
    assert result.is_valid
    assert result.validated_sql == sql
    assert result.error is None
