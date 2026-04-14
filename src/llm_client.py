from __future__ import annotations
import sqlparse
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
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
from openrouter.components import Reasoning
from openrouter import OpenRouter

import json
import time
from typing import Any

import structlog

from jinja2 import Environment
from src.config import PipelineConfig

# WHY: shared env so all templates inherit the same whitespace settings;
# trim_blocks removes the newline after a block tag,
# lstrip_blocks strips leading whitespace before block tags —
# together they produce clean output without manual {%- -%} stripping
_JINJA_ENV = Environment(trim_blocks=True, lstrip_blocks=True)

# ── SQL Generation ────────────────────────────────────────────────────────────

_SQL_GEN_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are an expert analytics SQL generator.
Your task is to convert a user question into correct SQLite SQL for a SINGLE TABLE database.

Rules:
* Output ONLY valid SQL
* Use SQLite syntax
* Use ONLY the provided table and columns — never invent columns
* Prefer explicit column names over SELECT *
* Use WHERE filters when implied
* Use GROUP BY only when needed
* Use ORDER BY only when useful
* Use LIMIT when returning individual rows
* SQLite date functions only: date(), datetime(), strftime()
* Never use ILIKE (not supported in SQLite)
* Handle NULL safely when relevant
* Read-only SQL only
* If the user's intent is to modify, delete, update, or insert data: answerable=false, sql=null
* If the question cannot be answered from the available schema and columns: answerable=false, sql=null\
""")

_SQL_GEN_USER_TMPL = _JINJA_ENV.from_string("""\
{% if ddl %}
SCHEMA:
{{ ddl }}

{% endif %}
{% if table_name %}
TABLE:
{{ table_name }}

{% endif %}
USER QUESTION:
{{ question }}
{% if correction_hint %}

CORRECTION CONTEXT:
{{ correction_hint }}
{% endif %}\
""")

# ── Answer Generation ─────────────────────────────────────────────────────────

_ANSWER_GEN_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are a data analyst assistant.
Answer the user's question using ONLY the SQL query results.

Rules:
* Use ONLY provided rows
* Never invent facts
* Use exact values from the data
* Explain trends only if visible in the rows
* If result is partial or limited, say so
* Write for a business user
* Be concise\
""")

_ANSWER_GEN_USER_TMPL = _JINJA_ENV.from_string("""\
USER QUESTION:
{{ question }}

SQL:
{{ sql }}
{% if columns %}

COLUMNS:
{{ columns }}
{% endif %}

ROWS:
{{ rows }}

ROW COUNT:
{{ count }}\
{% if correction_hint %}

CORRECTION CONTEXT:
{{ correction_hint }}
{% endif %}""")

# ── SQL Analytics Judge ───────────────────────────────────────────────────────

_SQL_JUDGE_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are a SQL quality reviewer for a SQLite analytics database.
Evaluate whether the SQL correctly answers the user request.

Rubric:
fail: wrong columns, wrong filters, wrong aggregation, non-executable SQL, columns not in schema, non-SELECT statement
borderline: mostly correct but incomplete, minor logic issue, missing filter, unnecessary columns selected
pass: correct, uses valid schema, appropriate logic, SELECT only, directly answers the question

Check: intent match, column validity, filter correctness, aggregation correctness, SQLite syntax\
""")

_SQL_JUDGE_USER_TMPL = _JINJA_ENV.from_string("""\
{% if ddl %}
SCHEMA:
{{ ddl }}

{% endif %}
QUESTION:
{{ question }}

SQL:
{{ sql }}\
""")

# ── Answer Grounding Judge ────────────────────────────────────────────────────

_GROUNDING_JUDGE_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are a strict evaluator of answers generated from SQL results.

Rubric:
fail: contradicts rows, wrong numbers, hallucinated claims, does not answer question
borderline: mostly correct but incomplete, unclear wording, misses an important detail
pass: correct, grounded in rows, clear, useful, answers question\
""")

_GROUNDING_JUDGE_USER_TMPL = _JINJA_ENV.from_string("""\
QUESTION:
{{ question }}

ROWS:
{{ rows }}

ANSWER:
{{ answer }}\
""")


logger = structlog.get_logger()


class LLMTokenLimitError(RuntimeError):
    """Raised when the LLM truncates its response because it hit the token cap.

    WHY: a distinct exception type lets callers (and _is_retryable_llm_error)
    distinguish token-limit exhaustion from transient transport errors.
    Retrying with the same parameters would produce the same truncated output,
    so this error must NOT be retried — only the signal should be logged.
    """


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for transient errors worth retrying; False for permanent failures.

    WHY: non-retryable errors (auth, bad request) will fail identically on every
    retry attempt, wasting quota and adding latency. Only retry errors that can
    resolve on their own (rate limits, transient server failures).
    LLMTokenLimitError is explicitly excluded — retrying with the same parameters
    would produce the same truncated output every time.
    """
    if isinstance(exc, LLMTokenLimitError):
        return False
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "500", "503", "rate limit", "overloaded"))


class OpenRouterLLMClient:
    """LLM client using the OpenRouter SDK for chat completions."""

    provider_name = "openrouter"

    def __init__(self, config: PipelineConfig) -> None:
        self.model = config.openrouter_model
        self._client = OpenRouter(api_key=config.openrouter_api_key)
        self._stats = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # WHY: LLM inference can take 5–30s; no timeout means a hung call blocks
        # the pipeline indefinitely
        self._timeout = config.llm_timeout_seconds
        # WHY: configurable so callers can reduce retries in test/latency-sensitive
        # contexts without changing code
        self._max_retries = config.llm_max_retries
        # WHY: store generation params at init so every call site uses consistent
        # values from config rather than scattered hardcoded literals
        self._sql_max_tokens = config.sql_max_tokens
        self._answer_max_tokens = config.answer_max_tokens
        self._sql_judge_max_tokens = config.sql_judge_max_tokens
        self._answer_judge_max_tokens = config.answer_judge_max_tokens
        self._sql_temperature = config.sql_temperature
        self._answer_temperature = config.answer_temperature
        self._sql_judge_temperature = config.sql_judge_temperature
        self._answer_judge_temperature = config.answer_judge_temperature
        self._answer_rows_sample = config.answer_rows_sample
        self._sql_reasoning_effort = config.sql_reasoning_effort
        self._answer_reasoning_effort = config.answer_reasoning_effort
        self._sql_judge_reasoning_effort = config.sql_judge_reasoning_effort
        self._answer_judge_reasoning_effort = config.answer_judge_reasoning_effort

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format=None,
        reasoning_effort: str | None = None,
    ) -> str:
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        # WHY: reasoning is only injected when explicitly set — omitting it lets
        # non-reasoning models ignore the parameter without breaking the call
        if reasoning_effort is not None:
            kwargs["reasoning"] = Reasoning(effort=reasoning_effort)

        # WHY: wrap only the SDK call (not the full method) so that stats
        # increment and response parsing happen exactly once on success, not
        # once per retry attempt. Exponential backoff with jitter avoids
        # thundering-herd effects when multiple calls retry simultaneously.
        @retry(
            retry=retry_if_exception(_is_retryable_llm_error),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _send():
            return self._client.chat.send(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                timeout_ms=self._timeout * 1000,
                **kwargs,
            )

        res = _send()

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

        # WHY: finish_reason "length" means the model hit its token cap and the
        # response is truncated. Retrying with the same parameters would produce
        # the same truncated output — raise a distinct non-retryable error so
        # callers get a clear signal instead of a confusing JSON parse failure.
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason == "length":
            raise LLMTokenLimitError(
                "LLM response truncated (finish_reason=length): the model hit its token cap. Increase max_tokens or shorten the prompt."
            )

        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter response content is not text.")
        return content.strip()

    # ── Prompt builders ───────────────────────────────────────────────────────
    # WHY: static methods keep prompt construction pure and testable without
    # needing an LLM client instance or mocking any HTTP calls

    @staticmethod
    def _build_sql_generation_messages(question: str, context: dict) -> list[dict]:
        # WHY: only ddl/table_name go to the LLM — columns/column_types are for
        # rule-based validation only and add noise to the prompt.
        # table_name derived from existing tables set — no new context key needed.
        tables = context.get("tables", set())
        table_name = next(iter(tables), "")
        return [
            {"role": "system", "content": _SQL_GEN_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _SQL_GEN_USER_TMPL.render(
                    ddl=context.get("ddl", ""),
                    table_name=table_name,
                    question=question,
                    correction_hint=context.get("correction_hint", ""),
                ),
            },
        ]

    @staticmethod
    def _build_answer_generation_messages(
        question: str,
        sql: str,
        rows: list[dict],
        correction_hint: str = "",
        rows_sample: int = 30,
    ) -> list[dict]:
        rows_data = rows[:rows_sample]
        # WHY: columns listed separately so the model reasons about result shape
        # even when rows are sparse or contain null values
        columns = list(rows_data[0].keys()) if rows_data else []
        return [
            {"role": "system", "content": _ANSWER_GEN_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _ANSWER_GEN_USER_TMPL.render(
                    question=question,
                    sql=sql,
                    columns=columns,
                    rows=json.dumps(rows_data, ensure_ascii=True),
                    count=len(rows),
                    correction_hint=correction_hint,
                ),
            },
        ]

    @staticmethod
    def _build_sql_judge_messages(
        question: str,
        sql: str,
        schema_context: dict,
    ) -> list[dict]:
        return [
            {"role": "system", "content": _SQL_JUDGE_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _SQL_JUDGE_USER_TMPL.render(
                    ddl=schema_context.get("ddl", ""),
                    question=question,
                    sql=sql,
                ),
            },
        ]

    @staticmethod
    def _build_grounding_judge_messages(
        question: str,
        rows: list[dict],
        answer: str,
    ) -> list[dict]:
        # WHY: SQL intentionally excluded — grounding judge verifies answer vs
        # rows only; including SQL risks evaluating SQL correctness instead.
        # No row cap — the judge needs the full result set to accurately detect
        # hallucinated values or missing data.
        return [
            {"role": "system", "content": _GROUNDING_JUDGE_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _GROUNDING_JUDGE_USER_TMPL.render(
                    question=question,
                    rows=json.dumps(rows, ensure_ascii=True),
                    answer=answer,
                ),
            },
        ]

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        logger.debug("SQL generation started", question_length=len(question))
        start = time.perf_counter()
        error = None
        sql = None

        try:
            text = self._chat(
                messages=self._build_sql_generation_messages(question, context),
                temperature=self._sql_temperature,
                max_tokens=self._sql_max_tokens,
                reasoning_effort=self._sql_reasoning_effort,
                response_format=ChatFormatJSONSchemaConfig(
                    type="json_schema",
                    json_schema=ChatJSONSchemaConfig(
                        name="sql_output",
                        schema_=SQLResponse.model_json_schema(),
                        strict=True,
                    ),
                ),
            )
            parsed = SQLResponse.model_validate_json(text)

            sql = parsed.sql
            if sql:
                try:
                    formatted_sql = sqlparse.format(
                        sql,
                        reindent=True,
                        keyword_case="upper",
                        indent_width=4,
                        wrap_after=80,
                        comma_first=False,
                        strip_comments=False,
                    )

                    sql = formatted_sql
                except Exception:
                    # Fallback if the parser fails on complex/invalid syntax
                    sql = sql.strip()

        except Exception as exc:
            logger.exception("SQL generation failed with an unexpected exception")
            error = str(exc)

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        logger.debug(
            "SQL generation completed",
            timing_ms=timing_ms,
            has_sql=sql is not None,
            has_error=error is not None,
        )
        return SQLGenerationOutput(
            sql=parsed.sql if parsed.answerable else None,
            answerable=parsed.answerable,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    def generate_answer(
        self, question: str, sql: str | None, rows: list[dict[str, Any]], correction_hint: str = ""
    ) -> AnswerGenerationOutput:
        logger.debug("Answer generation started", question_length=len(question), row_count=len(rows), has_sql=sql is not None)
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

        start = time.perf_counter()
        error = None
        answer = ""

        try:
            answer = self._chat(
                messages=self._build_answer_generation_messages(
                    question, sql, rows, correction_hint=correction_hint, rows_sample=self._answer_rows_sample
                ),
                temperature=self._answer_temperature,
                max_tokens=self._answer_max_tokens,
                reasoning_effort=self._answer_reasoning_effort,
            )
        except Exception as exc:
            logger.exception("Answer generation failed with an unexpected exception")
            error = str(exc)
            answer = f"Error generating answer: {error}"

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        logger.debug(
            "Answer generation completed",
            timing_ms=timing_ms,
            has_error=error is not None,
        )
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
        logger.debug("SQL analytics judge started", sql_preview=sql[:120])
        start = time.perf_counter()
        try:
            text = self._chat(
                messages=self._build_sql_judge_messages(question, sql, schema_context),
                temperature=self._sql_judge_temperature,
                max_tokens=self._sql_judge_max_tokens,
                reasoning_effort=self._sql_judge_reasoning_effort,
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
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "SQL analytics judge completed",
                verdict=parsed.verdict,
                grade=parsed.grade,
                timing_ms=timing_ms,
            )
            return SQLAnalyticsJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.exception("SQL analytics judge failed")
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
        logger.debug("Answer grounding judge started", answer_length=len(answer), row_count=len(rows))
        start = time.perf_counter()
        try:
            text = self._chat(
                messages=self._build_grounding_judge_messages(question, rows, answer),
                temperature=self._answer_judge_temperature,
                max_tokens=self._answer_judge_max_tokens,
                reasoning_effort=self._answer_judge_reasoning_effort,
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
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "Answer grounding judge completed",
                verdict=parsed.verdict,
                grade=parsed.grade,
                timing_ms=timing_ms,
            )
            return AnswerGroundingJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.exception("Answer grounding judge failed")
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


def build_default_llm_client(config: PipelineConfig) -> OpenRouterLLMClient:
    return OpenRouterLLMClient(config=config)
