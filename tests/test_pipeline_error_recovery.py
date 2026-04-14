"""Tests for edge case handling and error recovery in the pipeline.

Covers:
- Phase 1: Stage errors consistently mapped to PipelineOutput.status
- Phase 5a/5b/5d: Unified SQL correction loop (execution, analytics-judge, result-validation)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from helpers import BaseLLMStub, _ZERO_STATS
    from src.config import PipelineConfig
    from src.pipeline import AnalyticsPipeline
    from src.types import (
        AnswerGenerationOutput,
        SQLAnalyticsJudgeOutput,
        SQLExecutionOutput,
        SQLGenerationOutput,
    )
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zero_stats() -> dict[str, Any]:
    return dict(_ZERO_STATS)


def _make_pipeline(llm, db, metadata_db) -> AnalyticsPipeline:
    cfg = PipelineConfig(openrouter_api_key="test-key", db_path=db, metadata_db_path=metadata_db)
    return AnalyticsPipeline(config=cfg, llm_client=llm)


# ---------------------------------------------------------------------------
# Phase 1 — status determination covers answer generation errors
# ---------------------------------------------------------------------------


class _LLMAnswerError(BaseLLMStub):
    """Stub: generate_answer always returns an error."""

    def generate_answer(self, question, sql, rows, correction_hint="", conversation_context=""):
        return AnswerGenerationOutput(
            answer="Error generating answer: boom",
            timing_ms=0.0,
            llm_stats=_zero_stats(),
            error="boom",
        )


class _LLMExecutorAndAnswerError(BaseLLMStub):
    """Stub: generate_answer returns an error (executor also errors — execution wins)."""

    def generate_answer(self, question, sql, rows, correction_hint="", conversation_context=""):
        return AnswerGenerationOutput(
            answer="Error: answer error",
            timing_ms=0.0,
            llm_stats=_zero_stats(),
            error="answer error",
        )


class _StubExecutorError:
    """Minimal executor stub that always returns a DB error."""

    def run(self, sql):
        return SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error="db error")


def test_answer_gen_error_sets_status_error(analytics_db_with_data, schema_description_db):
    """When generate_answer fails, status must be 'error', not 'success'."""
    pipeline = _make_pipeline(_LLMAnswerError(), analytics_db_with_data, schema_description_db)
    output = pipeline.run("What is the average age?")
    assert output.status == "error"
    assert output.answer_generation.error == "boom"


def test_execution_error_takes_priority_over_answer_error(analytics_db, schema_description_db, monkeypatch):
    """Execution error governs status even if answer gen also errors."""
    pipeline = _make_pipeline(_LLMExecutorAndAnswerError(), analytics_db, schema_description_db)
    # Swap executor for one that always errors
    pipeline.executor = _StubExecutorError()
    output = pipeline.run("What is the average age?")
    assert output.status == "error"
    # Execution error is the governing one
    assert output.sql_execution.error == "db error"


def test_all_success_status_unaffected(analytics_db_with_data, schema_description_db):
    """Normal run (no errors anywhere) must still yield status='success'."""
    pipeline = _make_pipeline(BaseLLMStub(), analytics_db_with_data, schema_description_db)
    output = pipeline.run("What is the average age?")
    assert output.status == "success"


# ---------------------------------------------------------------------------
# Execution-guided SQL correction loop
# ---------------------------------------------------------------------------


class _ExecutorFailThenSucceed:
    """Executor that fails on first call, executes real SQL on second."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._call_count = 0

    def run(self, sql):
        self._call_count += 1
        if self._call_count == 1:
            return SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error="no such column: x")
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchmany(100)]
            return SQLExecutionOutput(rows=rows, row_count=len(rows), timing_ms=0.0)
        except Exception as exc:
            return SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error=str(exc))


class _LLMWithCorrection(BaseLLMStub):
    """Stub that returns a fixed corrected SQL on correction calls; records hints."""

    def __init__(self, corrected_sql: str):
        self._corrected_sql = corrected_sql
        self.correction_hints: list[dict] = []

    def generate_sql(self, question, context):
        if "correction_hint" in context:
            self.correction_hints.append({"hint": context["correction_hint"]})
            return SQLGenerationOutput(
                sql=self._corrected_sql,
                answerable=True,
                timing_ms=0.0,
                llm_stats=_zero_stats(),
            )
        return super().generate_sql(question, context)


def test_correction_recovers_on_execution_error(analytics_db_with_data, schema_description_db, monkeypatch):
    """When SQL execution errors and SQL_CORRECTION_ENABLED=true, corrected SQL is used."""
    monkeypatch.setenv("SQL_CORRECTION_ENABLED", "true")
    llm = _LLMWithCorrection("SELECT age FROM gaming_mental_health")
    pipeline = _make_pipeline(llm, analytics_db_with_data, schema_description_db)
    pipeline.executor = _ExecutorFailThenSucceed(analytics_db_with_data)
    output = pipeline.run("What are the ages?")
    assert output.status == "success"
    assert any(d.get("stage") == "sql_correction" for d in output.sql_generation.intermediate_outputs)


def test_correction_bounded_by_max_retries(analytics_db, schema_description_db, monkeypatch):
    """Correction loop stops after MAX_SQL_CORRECTION_RETRIES attempts."""
    monkeypatch.setenv("SQL_CORRECTION_ENABLED", "true")
    monkeypatch.setenv("MAX_SQL_CORRECTION_RETRIES", "2")
    call_count = 0

    class _AlwaysErrorExecutor:
        def run(self, sql):
            nonlocal call_count
            call_count += 1
            return SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error="always fails")

    llm = _LLMWithCorrection("SELECT age FROM gaming_mental_health")
    pipeline = _make_pipeline(llm, analytics_db, schema_description_db)
    pipeline.executor = _AlwaysErrorExecutor()
    output = pipeline.run("What are the ages?")
    assert call_count == 3  # 1 original + 2 correction retries
    assert output.status == "error"




# ---------------------------------------------------------------------------
# History accumulation — execution correction
# ---------------------------------------------------------------------------


class _AlwaysErrorExecutorStub:
    """Module-level executor stub that always returns a DB error."""

    def run(self, sql):
        return SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error="always fails")


class _LLMInvalidThenValid(BaseLLMStub):
    """generate_sql returns a non-SELECT statement initially; on correction returns valid SELECT."""

    def generate_sql(self, question, context):
        if "correction_hint" in context:
            return SQLGenerationOutput(
                sql="SELECT age FROM gaming_mental_health",
                answerable=True,
                timing_ms=0.0,
                llm_stats=_zero_stats(),
            )
        return SQLGenerationOutput(
            sql="DELETE FROM gaming_mental_health",
            answerable=True,
            timing_ms=0.0,
            llm_stats=_zero_stats(),
        )


def test_execution_correction_hint_accumulates_history(analytics_db, schema_description_db, monkeypatch):
    """On the 2nd correction call the hint must include both prior sql+error pairs."""
    monkeypatch.setenv("SQL_CORRECTION_ENABLED", "true")
    monkeypatch.setenv("MAX_SQL_CORRECTION_RETRIES", "3")
    llm = _LLMWithCorrection("SELECT age FROM gaming_mental_health")
    pipeline = _make_pipeline(llm, analytics_db, schema_description_db)
    pipeline.executor = _AlwaysErrorExecutorStub()
    pipeline.run("What are the ages?")

    assert len(llm.correction_hints) >= 2
    second_hint = llm.correction_hints[1]["hint"]
    assert "Previous failed attempts" in second_hint
    assert "  2." in second_hint  # WHY: history must list at least 2 attempts by the 2nd call


# ---------------------------------------------------------------------------
# Analytics judge correction
# ---------------------------------------------------------------------------


class _LLMAnalyticsJudgeAlwaysFail(BaseLLMStub):
    """judge_sql_analytics always returns verdict=False; generate_sql records correction hints."""

    def __init__(self):
        self.correction_hints: list[str] = []

    def generate_sql(self, question, context):
        if "correction_hint" in context:
            self.correction_hints.append(context["correction_hint"])
            return SQLGenerationOutput(
                sql="SELECT age FROM gaming_mental_health",
                answerable=True,
                timing_ms=0.0,
                llm_stats=_zero_stats(),
            )
        return super().generate_sql(question, context)

    def judge_sql_analytics(self, question, sql, schema_context):
        return SQLAnalyticsJudgeOutput(
            verdict=False, grade="fail", issues=["missing GROUP BY"], reason=""
        )


def test_analytics_judge_correction_triggered_on_fail(analytics_db_with_data, schema_description_db, monkeypatch):
    """When analytics judge fails and correction is enabled, correct_sql is called."""
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "true")
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_CORRECTION_ENABLED", "true")
    llm = _LLMAnalyticsJudgeAlwaysFail()
    pipeline = _make_pipeline(llm, analytics_db_with_data, schema_description_db)
    output = pipeline.run("What are the ages?")
    assert len(llm.correction_hints) > 0
    assert any(d.get("stage") == "sql_analytics_correction" for d in output.sql_generation.intermediate_outputs)



def test_analytics_judge_correction_bounded_by_max_retries(analytics_db_with_data, schema_description_db, monkeypatch):
    """Analytics correction stops after MAX_SQL_ANALYTICS_CORRECTION_RETRIES attempts."""
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "true")
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_CORRECTION_ENABLED", "true")
    monkeypatch.setenv("MAX_SQL_ANALYTICS_CORRECTION_RETRIES", "2")
    llm = _LLMAnalyticsJudgeAlwaysFail()
    pipeline = _make_pipeline(llm, analytics_db_with_data, schema_description_db)
    output = pipeline.run("What are the ages?")
    analytics_entries = [d for d in output.sql_generation.intermediate_outputs if d.get("stage") == "sql_analytics_correction"]
    assert len(analytics_entries) == 2


def test_analytics_correction_hint_accumulates_history(analytics_db_with_data, schema_description_db, monkeypatch):
    """On the 2nd analytics correction call the hint must include both prior sql+reason pairs."""
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "true")
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_CORRECTION_ENABLED", "true")
    monkeypatch.setenv("MAX_SQL_ANALYTICS_CORRECTION_RETRIES", "3")
    llm = _LLMAnalyticsJudgeAlwaysFail()
    pipeline = _make_pipeline(llm, analytics_db_with_data, schema_description_db)
    pipeline.run("What are the ages?")

    assert len(llm.correction_hints) >= 2
    second_hint = llm.correction_hints[1]
    assert "Previous failed attempts" in second_hint
    assert "  2." in second_hint  # WHY: history must list at least 2 attempts by the 2nd call


# ---------------------------------------------------------------------------
# Result validation correction
# ---------------------------------------------------------------------------


class _LLMResultValidationRecorder(BaseLLMStub):
    """generate_sql records correction hints and returns valid SQL; judge always passes."""

    def __init__(self):
        self.correction_hints: list[str] = []

    def generate_sql(self, question, context):
        if "correction_hint" in context:
            self.correction_hints.append(context["correction_hint"])
        return SQLGenerationOutput(
            sql="SELECT age FROM gaming_mental_health",
            answerable=True,
            timing_ms=0.0,
            llm_stats=_zero_stats(),
        )


def test_result_validation_correction_triggered_on_empty_result(analytics_db, schema_description_db, monkeypatch):
    """When result validation flags empty_result and correction is enabled, correct_sql is called."""
    monkeypatch.setenv("RESULT_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("RESULT_VALIDATION_CORRECTION_ENABLED", "true")
    llm = _LLMResultValidationRecorder()
    pipeline = _make_pipeline(llm, analytics_db, schema_description_db)
    output = pipeline.run("What are the ages?")
    assert len(llm.correction_hints) > 0
    assert any(d.get("stage") == "sql_result_validation_correction" for d in output.sql_generation.intermediate_outputs)



def test_result_validation_correction_bounded_by_max_retries(analytics_db, schema_description_db, monkeypatch):
    """Result validation correction stops after MAX_RESULT_VALIDATION_CORRECTION_RETRIES attempts."""
    monkeypatch.setenv("RESULT_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("RESULT_VALIDATION_CORRECTION_ENABLED", "true")
    monkeypatch.setenv("MAX_RESULT_VALIDATION_CORRECTION_RETRIES", "2")
    llm = _LLMResultValidationRecorder()
    pipeline = _make_pipeline(llm, analytics_db, schema_description_db)
    output = pipeline.run("What are the ages?")
    result_entries = [d for d in output.sql_generation.intermediate_outputs if d.get("stage") == "sql_result_validation_correction"]
    assert len(result_entries) == 2


# ---------------------------------------------------------------------------
# Validation failure correction
# ---------------------------------------------------------------------------


def test_correction_recovers_on_validation_error(analytics_db_with_data, schema_description_db, monkeypatch):
    """When SQL fails validation and SQL_CORRECTION_ENABLED=true, corrected SQL is used."""
    monkeypatch.setenv("SQL_CORRECTION_ENABLED", "true")
    llm = _LLMInvalidThenValid()
    pipeline = _make_pipeline(llm, analytics_db_with_data, schema_description_db)
    output = pipeline.run("What are the ages?")
    assert output.status == "success"
    assert any(d.get("stage") == "sql_correction" for d in output.sql_generation.intermediate_outputs)
