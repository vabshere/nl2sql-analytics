"""Shared test utilities for pipeline integration tests.

WHY: BaseLLMStub, _ZERO_STATS, and make_llm_client are imported explicitly by
test modules, not auto-injected by pytest — they belong here, not in conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.llm_client import OpenRouterLLMClient
    from src.types import (
        AnswerGenerationOutput,
        AnswerGroundingJudgeOutput,
        IntentClassificationOutput,
        SQLAnalyticsJudgeOutput,
        SQLGenerationOutput,
        SummarizationOutput,
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


def make_llm_client(
    *,
    timeout: int = 60,
    max_retries: int = 3,
    sql_max_tokens: int = 10000,
    answer_max_tokens: int = 220,
    sql_judge_max_tokens: int = 20000,
    answer_judge_max_tokens: int = 20000,
    sql_temperature: float = 0.0,
    answer_temperature: float = 0.2,
    sql_judge_temperature: float = 0.0,
    answer_judge_temperature: float = 0.0,
    answer_rows_sample: int = 30,
    sql_reasoning_effort: str | None = None,
    answer_reasoning_effort: str | None = None,
    sql_judge_reasoning_effort: str | None = None,
    answer_judge_reasoning_effort: str | None = None,
    intent_max_tokens: int = 200,
) -> OpenRouterLLMClient:
    """Build an OpenRouterLLMClient with a stubbed SDK, bypassing __init__.

    WHY: __new__ is used so tests that only exercise specific methods (judges,
    prompt builders) are not forced to construct a full PipelineConfig or hold
    a real API key. All config-derived instance attributes are set explicitly
    here so they match the defaults from PipelineConfig.
    """
    client = OpenRouterLLMClient.__new__(OpenRouterLLMClient)
    client.model = "stub"
    client._stats = dict(_ZERO_STATS)
    client._client = MagicMock()
    client._timeout = timeout
    client._max_retries = max_retries
    client._sql_max_tokens = sql_max_tokens
    client._answer_max_tokens = answer_max_tokens
    client._sql_judge_max_tokens = sql_judge_max_tokens
    client._answer_judge_max_tokens = answer_judge_max_tokens
    client._sql_temperature = sql_temperature
    client._answer_temperature = answer_temperature
    client._sql_judge_temperature = sql_judge_temperature
    client._answer_judge_temperature = answer_judge_temperature
    client._answer_rows_sample = answer_rows_sample
    client._sql_reasoning_effort = sql_reasoning_effort
    client._answer_reasoning_effort = answer_reasoning_effort
    client._sql_judge_reasoning_effort = sql_judge_reasoning_effort
    client._answer_judge_reasoning_effort = answer_judge_reasoning_effort
    client._intent_max_tokens = intent_max_tokens
    return client


class BaseLLMStub:
    """Base stub LLM for pipeline integration tests.

    Provides no-op implementations of all LLM interface methods so each
    subclass only needs to override the method it specifically tests.
    """

    model = "stub"

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        return SQLGenerationOutput(
            sql="SELECT age FROM gaming_mental_health",
            answerable=True,
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def generate_answer(
        self,
        question: str,
        sql: str | None,
        rows: list,
        correction_hint: str = "",
        conversation_context: str = "",
    ) -> AnswerGenerationOutput:
        return AnswerGenerationOutput(
            answer="stub answer",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def judge_sql_analytics(self, question: str, sql: str, schema_context: dict) -> SQLAnalyticsJudgeOutput:
        return SQLAnalyticsJudgeOutput(verdict=True, grade="pass", issues=[], reason="")

    def judge_answer_grounding(self, question: str, sql: str, rows: list, answer: str) -> AnswerGroundingJudgeOutput:
        return AnswerGroundingJudgeOutput(verdict=True, grade="pass", issues=[], reason="")

    def correct_sql(self, question: str, failed_sql: str, db_error: str, context: dict) -> SQLGenerationOutput:
        # WHY: delegates to generate_sql so stubs that don't override this still
        # return a valid SQLGenerationOutput without duplicating stub machinery
        return self.generate_sql(question, context)

    def summarize_turns(self, turns: list) -> SummarizationOutput:
        return SummarizationOutput(summary="stub summary", llm_stats=dict(_ZERO_STATS))

    def classify_intent(self, question: str, conversation: Any) -> IntentClassificationOutput:
        return IntentClassificationOutput(
            intent="follow_up", reason="stub", llm_stats=dict(_ZERO_STATS)
        )

    def answer_from_context(self, question: str, conversation_context: str) -> AnswerGenerationOutput:
        return AnswerGenerationOutput(answer="stub context answer", timing_ms=0.0, llm_stats=dict(_ZERO_STATS))

    def pop_stats(self) -> dict[str, Any]:
        return dict(_ZERO_STATS)
