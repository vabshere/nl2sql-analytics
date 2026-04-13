from __future__ import annotations
from src.types import (
    AnswerGenerationOutput,
    AnswerGroundingJudgeOutput,
    JudgeResponse,
    SQLAnalyticsJudgeOutput,
    SQLGenerationOutput,
    SQLResponse,
)
from openrouter.components.chatformatjsonschemaconfig import (
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
)

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "openai/gpt-5-nano"


class OpenRouterLLMClient:
    """LLM client using the OpenRouter SDK for chat completions."""

    provider_name = "openrouter"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        try:
            from openrouter import OpenRouter
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: install 'openrouter'.") from exc
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self._client = OpenRouter(api_key=api_key)
        self._stats = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format=None,
    ) -> str:
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format

        res = self._client.chat.send(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )

        # WHY: llm_calls is always incremented regardless of whether usage is
        # present — some models/providers may return None usage on errors.
        self._stats["llm_calls"] += 1
        usage = getattr(res, "usage", None)
        if usage is not None:
            # WHY: SDK returns float token counts (e.g. 557.0); cast to int so
            # stats are always integers rather than floats
            self._stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self._stats["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

        choices = getattr(res, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter response contained no choices.")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter response content is not text.")
        return content.strip()

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        # WHY: structured output guarantees valid JSON; non-JSON is an unexpected
        # API failure that should surface as an error, not be silently swallowed
        parsed = json.loads(text)
        sql = parsed.get("sql") if isinstance(parsed, dict) else None
        if isinstance(sql, str) and sql.strip():
            return sql.strip()
        return None

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        system_prompt = (
            "You are a SQL assistant. "
            "Generate SQLite SELECT queries from natural language questions. "
            "Return your response in a format that can be parsed to extract the SQL."
        )
        user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nGenerate a SQL query to answer this question."

        start = time.perf_counter()
        error = None
        sql = None

        try:
            text = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=10000,
                response_format=ChatFormatJSONSchemaConfig(
                    type="json_schema",
                    json_schema=ChatJSONSchemaConfig(
                        name="sql_output",
                        schema_=SQLResponse.model_json_schema(),
                        strict=True,
                    ),
                ),
            )
            sql = self._extract_sql(text)
        except Exception as exc:
            error = str(exc)

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        return SQLGenerationOutput(
            sql=sql,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    def generate_answer(self, question: str, sql: str | None, rows: list[dict[str, Any]]) -> AnswerGenerationOutput:
        if not sql:
            return AnswerGenerationOutput(
                answer="I cannot answer this with the available table and schema. Please rephrase using known survey fields.",
                timing_ms=0.0,
                llm_stats={
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "model": self.model,
                },
                error=None,
            )
        if not rows:
            return AnswerGenerationOutput(
                answer="Query executed, but no rows were returned.",
                timing_ms=0.0,
                llm_stats={
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "model": self.model,
                },
                error=None,
            )

        system_prompt = "You are a concise analytics assistant. Use only the provided SQL results. Do not invent data."
        user_prompt = (
            f"Question:\n{question}\n\nSQL:\n{sql}\n\n"
            f"Rows (JSON):\n{json.dumps(rows[:30], ensure_ascii=True)}\n\n"
            "Write a concise answer in plain English."
        )

        start = time.perf_counter()
        error = None
        answer = ""

        try:
            answer = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=220,
            )
        except Exception as exc:
            error = str(exc)
            answer = f"Error generating answer: {error}"

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        return AnswerGenerationOutput(
            answer=answer,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    def judge_sql_analytics(
        self,
        question: str,
        sql: str,
        schema_context: dict,
    ) -> SQLAnalyticsJudgeOutput:
        """Evaluate whether the SQL is analytically correct for the question.

        WHY: rule-based validators check SQL structure; this checks analytical
        intent — wrong aggregation function, missing GROUP BY, wrong granularity.
        Non-blocking: result is stored in intermediate_outputs only.

        Returns a dict ready to append to SQLGenerationOutput.intermediate_outputs.
        """
        system_prompt = (
            "You are an expert analytics SQL reviewer. "
            "Evaluate whether a SQL query correctly implements the analytical intent "
            "of a natural language question. "
            "Consider: (1) aggregation — does the query aggregate when the question "
            "asks for a summary (average, count, total)? (2) grouping — does GROUP BY "
            "match the breakdown dimension in the question? (3) filtering — is the "
            "WHERE clause consistent with the question's scope? (4) granularity — "
            "does the result shape match what the question expects?"
        )
        user_prompt = f"Question: {question}\n\nSQL:\n{sql}\n\nSchema:\n{schema_context.get('ddl', '')}"

        try:
            text = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=20000,
                response_format=ChatFormatJSONSchemaConfig(
                    type="json_schema",
                    json_schema=ChatJSONSchemaConfig(
                        name="sql_analytics_judge_output",
                        schema_=JudgeResponse.model_json_schema(),
                        strict=True,
                    ),
                ),
            )
            parsed = JudgeResponse.model_validate_json(text)
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return SQLAnalyticsJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.warning("sql_analytics_judge failed: %s", exc)
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return SQLAnalyticsJudgeOutput(verdict=False, grade="fail", issues=[], reason="", error=str(exc), llm_stats=llm_stats)

    def judge_answer_grounding(
        self,
        question: str,
        sql: str,
        rows: list[dict[str, Any]],
        answer: str,
    ) -> AnswerGroundingJudgeOutput:
        """Evaluate whether the answer is grounded in the SQL results.

        WHY: generate_answer() uses a constrained system prompt but cannot
        self-verify. This provides an independent post-generation check.
        Non-blocking: result stored in intermediate_outputs only.

        Returns a dict ready to append to AnswerGenerationOutput.intermediate_outputs.
        """
        system_prompt = (
            "You are an answer quality auditor for a data analytics system. "
            "Verify that the answer is grounded in the provided data rows — "
            "it should not state values, entities, or conclusions not present in the rows."
        )
        # WHY: include only result rows (not full schema) — Arize AI research shows
        # judge performance improves when limited to data actually in scope
        rows_sample = json.dumps(rows[:30], ensure_ascii=True)
        user_prompt = f"Question: {question}\n\nSQL:\n{sql}\n\nData rows (up to 10):\n{rows_sample}\n\nAnswer to verify:\n{answer}"

        try:
            text = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=20000,
                response_format=ChatFormatJSONSchemaConfig(
                    type="json_schema",
                    json_schema=ChatJSONSchemaConfig(
                        name="answer_grounding_judge_output",
                        schema_=JudgeResponse.model_json_schema(),
                        strict=True,
                    ),
                ),
            )
            parsed = JudgeResponse.model_validate_json(text)
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return AnswerGroundingJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.warning("answer_grounding_judge failed: %s", exc)
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return AnswerGroundingJudgeOutput(verdict=False, grade="fail", issues=[], reason="", error=str(exc), llm_stats=llm_stats)

    def pop_stats(self) -> dict[str, Any]:
        out = dict(self._stats or {})
        self._stats = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return out


def build_default_llm_client() -> OpenRouterLLMClient:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    return OpenRouterLLMClient(api_key=api_key)
