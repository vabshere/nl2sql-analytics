from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.schema_context import (
        build_ddl,
        introspect_columns,
        load_descriptions,
        load_schema_context,
    )
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc

TABLE_NAME = "gaming_mental_health"
MISSING_DB = PROJECT_ROOT / "data" / "does_not_exist.sqlite"

# Minimal column set for pure unit tests (no DB needed)
SAMPLE_COLUMNS = [
    ("age", "INTEGER"),
    ("gender", "TEXT"),
    ("unknown_col", "REAL"),
]
SAMPLE_DESCRIPTIONS = {
    "age": {"description": "Age of the participant", "domain": "Demographic"},
    "gender": {"description": "Gender category", "domain": "Demographic"},
}


# ---------------------------------------------------------------------------
# load_descriptions
# ---------------------------------------------------------------------------


def test_load_descriptions(schema_description_db):
    table_desc, col_descs = load_descriptions(schema_description_db, TABLE_NAME)
    assert table_desc == "Gaming and Mental Health Behavioral Dataset"
    # WHY: mock DB has exactly 2 columns (age, gender)
    assert len(col_descs) == 2
    assert col_descs["age"]["description"] == "Age of the participant"
    assert col_descs["age"]["domain"] == "Demographic"


# ---------------------------------------------------------------------------
# build_ddl — pure unit tests (no DB required)
# ---------------------------------------------------------------------------


def test_build_ddl_without_descriptions():
    ddl = build_ddl(SAMPLE_COLUMNS, TABLE_NAME, "", {}, include_description=False)
    assert ddl.strip().startswith(f"CREATE TABLE {TABLE_NAME} (")
    assert ddl.strip().endswith(");")
    assert "--" not in ddl
    for name, col_type in SAMPLE_COLUMNS:
        assert name in ddl
        assert col_type in ddl
    # last column line must not have a trailing comma
    lines = [l.rstrip() for l in ddl.splitlines() if l.strip() and l.strip() != ");"]
    assert not lines[-1].endswith(",")
    # column order preserved
    assert ddl.index("age") < ddl.index("gender") < ddl.index("unknown_col")


def test_build_ddl_with_descriptions():
    ddl = build_ddl(
        SAMPLE_COLUMNS,
        TABLE_NAME,
        "Gaming and Mental Health Behavioral Dataset",
        SAMPLE_DESCRIPTIONS,
        include_description=True,
    )
    assert "-- Gaming and Mental Health Behavioral Dataset" in ddl
    assert "-- Age of the participant" in ddl
    assert "-- Gender category" in ddl
    # unknown_col has no description — must appear without a comment and not be dropped
    unknown_line = next(l for l in ddl.splitlines() if "unknown_col" in l)
    assert "--" not in unknown_line


# ---------------------------------------------------------------------------
# introspect_columns
# ---------------------------------------------------------------------------


def test_introspect_columns(analytics_db):
    cols = introspect_columns(analytics_db, TABLE_NAME)
    # WHY: mock DB has exactly 3 columns (age, gender, playtime_hours)
    assert len(cols) == 3
    for name, col_type in cols:
        assert isinstance(name, str) and name
        assert isinstance(col_type, str) and col_type
    # age is the first column in the mock CREATE TABLE
    assert cols[0][0] == "age"


# ---------------------------------------------------------------------------
# load_schema_context
# ---------------------------------------------------------------------------


def test_load_schema_context_default(analytics_db, schema_description_db):
    result = load_schema_context(analytics_db, schema_description_db, TABLE_NAME, include_description=True)
    assert {"ddl", "tables", "columns", "column_types"} <= set(result.keys())
    assert result["ddl"].strip().startswith(f"CREATE TABLE {TABLE_NAME}")
    # include_description=True → descriptions present
    assert "-- Age of the participant" in result["ddl"]
    # schema validation keys are populated
    assert TABLE_NAME in result["tables"]
    assert TABLE_NAME in result["columns"]
    assert TABLE_NAME in result["column_types"]


@pytest.mark.parametrize(
    "include_description,expected_in_ddl",
    [
        (True, "-- Age of the participant"),
        (False, None),
    ],
)
def test_load_schema_context_include_description_param(
    analytics_db, schema_description_db, include_description, expected_in_ddl
):
    result = load_schema_context(analytics_db, schema_description_db, TABLE_NAME, include_description=include_description)
    if expected_in_ddl:
        assert expected_in_ddl in result["ddl"]
    else:
        assert "--" not in result["ddl"]


def test_load_schema_context_missing_metadata_db_still_works(analytics_db):
    result = load_schema_context(analytics_db, MISSING_DB, TABLE_NAME, include_description=True)
    assert "ddl" in result
    assert "--" not in result["ddl"]
