"""Tests for edge case handling and error recovery in the pipeline.

Covers:
- Phase 1: Stage errors consistently mapped to PipelineOutput.status
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from helpers import BaseLLMStub, _ZERO_STATS
    from src.pipeline import AnalyticsPipeline
    from src.types import (
        AnswerGenerationOutput,
        SQLExecutionOutput,
    )
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zero_stats() -> dict[str, Any]:
    return dict(_ZERO_STATS)


def _make_pipeline(llm, db, metadata_db) -> AnalyticsPipeline:
    return AnalyticsPipeline(db_path=db, llm_client=llm, metadata_db_path=metadata_db)


# ---------------------------------------------------------------------------
# Phase 1 — status determination covers answer generation errors
# ---------------------------------------------------------------------------


class _LLMAnswerError(BaseLLMStub):
    """Stub: generate_answer always returns an error."""

    def generate_answer(self, question, sql, rows):
        return AnswerGenerationOutput(
            answer="Error generating answer: boom",
            timing_ms=0.0,
            llm_stats=_zero_stats(),
            error="boom",
        )


class _LLMExecutorAndAnswerError(BaseLLMStub):
    """Stub: generate_answer returns an error (executor also errors — execution wins)."""

    def generate_answer(self, question, sql, rows):
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
