"""Shared test utilities for pipeline integration tests.

WHY: BaseLLMStub and _ZERO_STATS are imported explicitly by test modules,
not auto-injected by pytest — they belong here, not in conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.types import (
        AnswerGenerationOutput,
        AnswerGroundingJudgeOutput,
        SQLAnalyticsJudgeOutput,
        SQLGenerationOutput,
    )
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc

_ZERO_STATS: dict[str, Any] = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "model": "stub",
}


class BaseLLMStub:
    """Base stub LLM for pipeline integration tests.

    Provides no-op implementations of all LLM interface methods so each
    subclass only needs to override the method it specifically tests.
    """

    model = "stub"

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        return SQLGenerationOutput(
            sql="SELECT age FROM gaming_mental_health",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def generate_answer(self, question: str, sql: str | None, rows: list) -> AnswerGenerationOutput:
        return AnswerGenerationOutput(
            answer="stub answer",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def judge_sql_analytics(self, question: str, sql: str, schema_context: dict) -> SQLAnalyticsJudgeOutput:
        return SQLAnalyticsJudgeOutput(verdict=True, grade="pass", issues=[], reason="")

    def judge_answer_grounding(self, question: str, sql: str, rows: list, answer: str) -> AnswerGroundingJudgeOutput:
        return AnswerGroundingJudgeOutput(verdict=True, grade="pass", issues=[], reason="")

    def pop_stats(self) -> dict[str, Any]:
        return dict(_ZERO_STATS)
