"""Type definitions for SQL Agent pipeline input/output structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class SQLResponse(BaseModel):
    """LLM response schema for SQL generation."""

    sql: str | None
    answerable: bool = Field(description="Whether answer can be queried.")


class JudgeResponse(BaseModel):
    """Common LLM response schema enforced via structured output for all judges."""

    verdict: bool = Field(description="True if acceptable; false if materially incorrect.")

    grade: Literal["fail", "borderline", "pass"] = Field(description="Overall quality classification.")

    issues: list[str] = Field(description="Specific problems found; empty list if none.")

    reason: str = Field(description="Brief explanation of the decision.")


class JudgeOutput(JudgeResponse):
    """Common base for typed judge outputs, extending the response with pipeline metadata.

    WHY: output inherits the full response schema; only new fields are added here —
    stage (which judge produced this) and error (set from caught exception on failure,
    None on success).
    """

    stage: str
    error: str | None = None
    llm_stats: dict[str, Any] = Field(default_factory=dict)


class SQLAnalyticsJudgeOutput(JudgeOutput):
    """Typed output for judge_sql_analytics()."""

    stage: Literal["sql_analytics_judge"] = "sql_analytics_judge"


class AnswerGroundingJudgeOutput(JudgeOutput):
    """Typed output for judge_answer_grounding()."""

    stage: Literal["answer_grounding_judge"] = "answer_grounding_judge"


@dataclass
class PipelineInput:
    """Input to the AnalyticsPipeline.run() method."""

    question: str
    request_id: str | None = None


@dataclass
class SQLGenerationOutput:
    """Output from the SQL generation stage.

    For complex solutions with multiple LLM calls (chain-of-thought, planning,
    query refinement), populate intermediate_outputs with per-call details.
    llm_stats aggregates all calls for efficient evaluation.
    """

    sql: str | None
    answerable: bool | None
    timing_ms: float
    # Aggregated: {llm_calls, prompt_tokens, completion_tokens, total_tokens, model}
    llm_stats: dict[str, Any]
    intermediate_outputs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class SQLValidationOutput:
    """Output from the SQL validation stage."""

    is_valid: bool
    validated_sql: str | None
    error: str | None = None
    timing_ms: float = 0.0


@dataclass
class SQLExecutionOutput:
    """Output from the SQL execution stage."""

    rows: list[dict[str, Any]]
    row_count: int
    timing_ms: float
    error: str | None = None
    # WHY: fetchmany(100) silently drops rows beyond the limit; this flag lets callers
    # detect truncation without a second query
    rows_truncated: bool = False


@dataclass
class AnswerGenerationOutput:
    """Output from the answer generation stage.

    For complex solutions with multiple LLM calls (summarization, verification),
    populate intermediate_outputs with per-call details.
    llm_stats aggregates all calls for efficient evaluation.
    """

    answer: str
    timing_ms: float
    # Aggregated: {llm_calls, prompt_tokens, completion_tokens, total_tokens, model}
    llm_stats: dict[str, Any]
    intermediate_outputs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class ResultValidationOutput:
    """Post-execution result shape signals. Always informational — never blocks the pipeline.

    WHY: empty_result is already handled in generate_answer(). These flags exist for
    monitoring, confidence scoring, and potential future use in a correction loop.
    """

    flags: list[str] = field(default_factory=list)
    # Possible values: "empty_result", "all_null_column:<col>", "all_zero_column:<col>"
    timing_ms: float = 0.0


@dataclass
class PipelineOutput:
    """Complete output from AnalyticsPipeline.run()."""

    # Status
    status: str  # "success" | "unanswerable" | "invalid_sql" | "error"
    question: str
    request_id: str | None

    # Stage outputs (for evaluation)
    sql_generation: SQLGenerationOutput
    sql_validation: SQLValidationOutput
    sql_execution: SQLExecutionOutput
    answer_generation: AnswerGenerationOutput

    # Convenience fields
    sql: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    # Aggregates
    timings: dict[str, float] = field(default_factory=dict)
    total_llm_stats: dict[str, Any] = field(default_factory=dict)

    # Post-execution result quality signals (informational, never blocks)
    result_validation: ResultValidationOutput = field(default_factory=ResultValidationOutput)
