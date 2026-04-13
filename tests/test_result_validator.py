from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from helpers import BaseLLMStub, _ZERO_STATS
    from src.pipeline import AnalyticsPipeline, ResultValidator
    from src.types import ResultValidationOutput, SQLExecutionOutput
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


class _StubLLM(BaseLLMStub):
    pass


def _exec(rows: list[dict], error: str | None = None) -> SQLExecutionOutput:
    return SQLExecutionOutput(rows=rows, row_count=len(rows), timing_ms=0.0, error=error)


# ---------------------------------------------------------------------------
# result flag checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,expected_flag",
    [
        # empty result → flagged
        ([], "empty_result"),
        # non-empty result → no flags
        ([{"age": 25, "gender": "Male"}], None),
        ([{"age": None}, {"age": None}, {"age": None}], "all_null_column:age"),  # all null → flagged
        # partial null → not flagged
        ([{"age": None}, {"age": 25}], None),
        ([{"score": 0}, {"score": 0.0}, {"score": 0}], "all_zero_column:score"),  # all numeric zero → flagged
        # partial zero → not flagged
        ([{"score": 0}, {"score": 5}], None),
        # text "0" is not numeric zero
        ([{"label": "0"}, {"label": "0"}], None),
    ],
)
def test_result_flags(rows, expected_flag):
    result = ResultValidator.validate(_exec(rows=rows))
    if expected_flag:
        assert expected_flag in result.flags
    else:
        assert result.flags == []


# ---------------------------------------------------------------------------
# execution error — skip checks
# ---------------------------------------------------------------------------


def test_execution_error_skips_checks():
    # Even with empty rows, if there's an execution error, skip all checks
    result = ResultValidator.validate(_exec(rows=[], error="no such table: foo"))
    assert result.flags == []


# ---------------------------------------------------------------------------
# env var gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_val,expect_flag",
    [
        ("false", False),
        ("true", True),
    ],
)
def test_env_var_gate(monkeypatch, env_val, expect_flag):
    monkeypatch.setenv("RESULT_VALIDATION_ENABLED", env_val)
    result = ResultValidator.validate(_exec(rows=[{"age": None}]))
    if expect_flag:
        assert "all_null_column:age" in result.flags
    else:
        assert result.flags == []
        assert result.timing_ms == 0.0


# ---------------------------------------------------------------------------
# pipeline wiring — stub end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_wires_result_validation(analytics_db, schema_description_db):
    pipeline = AnalyticsPipeline(
        db_path=analytics_db,
        llm_client=_StubLLM(),
        metadata_db_path=schema_description_db,
    )
    output = pipeline.run("any question")
    assert hasattr(output, "result_validation")
    assert isinstance(output.result_validation, ResultValidationOutput)
