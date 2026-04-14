from __future__ import annotations
import sqlparse
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from src.tracing import get_tracer
from src.types import (
    AnswerGenerationOutput,
    AnswerGroundingJudgeOutput,
    IntentClassificationOutput,
    JudgeResponse,
    SQLAnalyticsJudgeOutput,
    SQLGenerationOutput,
    SQLResponse,
    SummarizationOutput,
)
from openrouter.components.chatformatjsonschemaconfig import (
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
)
from openrouter.components import Reasoning
from openrouter import OpenRouter
from pydantic import BaseModel, Field

import json
import time
from typing import Any, Literal

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
{% if conversation_context %}
CONVERSATION CONTEXT:
{{ conversation_context }}

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
{% endif %}
{% if conversation_context %}

CONVERSATION CONTEXT:
{{ conversation_context }}
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


# ── Conversation history — Summarization ─────────────────────────────────────

_SUMMARIZE_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are summarizing a data analytics conversation for use as future context.
Write a concise bullet-point summary: what was asked, what the SQL queried, what the answer was.
Be brief — this summary will be prepended to future LLM prompts.\
""")

_SUMMARIZE_USER_TMPL = _JINJA_ENV.from_string("""\
TURNS TO SUMMARIZE:
{{ turns_text }}\
""")

# ── Conversation history — Intent Classification ──────────────────────────────


class _IntentResponse(BaseModel):
    """Structured output schema for intent classification.

    WHY: enforced via response_format=json_schema — same pattern as SQLResponse
    and JudgeResponse. Field descriptions guide the model without inline JSON
    instructions in the system prompt.
    """

    intent: Literal["new_query", "follow_up", "data_question"] = Field(
        description=(
            "follow_up: refines or extends a prior query; references prior filters, subsets, or metrics. "
            "data_question: asks about information already present in prior results — no new SQL needed. "
            "new_query: introduces a new subject, metric, or time range unrelated to history."
        )
    )
    reason: str = Field(description="Brief one-sentence explanation of the classification decision.")


# WHY: from __future__ import annotations postpones annotation evaluation; Pydantic
# cannot resolve Literal at class-definition time without an explicit rebuild call.
_IntentResponse.model_rebuild()

_INTENT_CLASSIF_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are an intent classifier for a data analytics assistant. Given a conversation history \
and a new question, classify the question into exactly one of three intents:

data_question  — The answer can be derived directly from result rows or values already \
present in the conversation history. No new database query is needed. \
Key signal: the question refers to specific values or rows already returned \
(e.g. "from the results you just showed", "of those", "which of those", \
"based on what you just calculated", "from the list above").

follow_up      — The question extends or refines the previous query topic but requires \
new data from the database (e.g. a related metric, a breakdown, or a more detailed view \
of the same subject).

new_query      — The question is about a completely different topic or metric with \
no meaningful connection to the prior conversation.

Decision rule: if the existing result rows in the history are sufficient to answer the \
question WITHOUT querying the database again, choose data_question. Otherwise, if the \
topic is related, choose follow_up. If unrelated, choose new_query.

Always populate the reason field with a one-sentence explanation before committing to an intent.\
""")

_INTENT_CLASSIF_USER_TMPL = _JINJA_ENV.from_string("""\
CONVERSATION HISTORY:
{{ history }}

NEW QUESTION:
{{ question }}\
""")

# ── Conversation history — Context Answer ─────────────────────────────────────

_CONTEXT_ANSWER_SYSTEM_TMPL = _JINJA_ENV.from_string("""\
You are a data analyst assistant. Answer the user's question using ONLY the conversation history provided.
The history contains prior questions, SQL queries, and answers from this session.
Use ONLY the information in the history — never invent new facts or run new analysis.\
""")

_CONTEXT_ANSWER_USER_TMPL = _JINJA_ENV.from_string("""\
CONVERSATION CONTEXT:
{{ conversation_context }}

QUESTION:
{{ question }}\
""")


logger = structlog.get_logger()

_tracer = get_tracer(__name__)


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
        self._intent_max_tokens = config.intent_max_tokens

    @_tracer.start_as_current_span("llm.chat")
    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format=None,
        reasoning_effort: str | None = None,
    ) -> str:
        span = trace.get_current_span()
        # WHY: GenAI semantic conventions — standard attributes for LLM calls
        # that Phoenix/OTEL backends understand and can aggregate
        span.set_attribute("gen_ai.system", "openrouter")
        span.set_attribute("gen_ai.request.model", self.model)
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("gen_ai.request.temperature", float(temperature))

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

        try:
            res = _send()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise

        # WHY: llm_calls is always incremented regardless of whether usage is
        # present — some models/providers may return None usage on errors.
        self._stats["llm_calls"] += 1
        usage = getattr(res, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            # WHY: SDK returns float token counts (e.g. 557.0); cast to int so
            # stats are always integers rather than floats
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self._stats["prompt_tokens"] += prompt_tokens
            self._stats["completion_tokens"] += completion_tokens
            self._stats["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        # WHY: set token counts on the span after accumulating so Phoenix can
        # display per-call usage without needing to drain self._stats
        span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

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
                    conversation_context=context.get("conversation_context", ""),
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
        conversation_context: str = "",
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
                    conversation_context=conversation_context,
                ),
            },
        ]

    @staticmethod
    def _build_summarization_messages(turns: list) -> list[dict]:
        """Format turns into a summarization prompt."""
        lines = []
        for i, turn in enumerate(turns, 1):
            sql_display = turn.sql if turn.sql is not None else "(none)"
            lines.append(f"Turn {i}:\n  Question: {turn.question}\n  SQL: {sql_display}\n  Answer: {turn.answer}")
        turns_text = "\n\n".join(lines)
        return [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_TMPL.render()},
            {"role": "user", "content": _SUMMARIZE_USER_TMPL.render(turns_text=turns_text)},
        ]

    @staticmethod
    def _build_intent_messages(question: str, conversation: Any) -> list[dict]:
        """Format conversation history and question into an intent classification prompt."""
        return [
            {"role": "system", "content": _INTENT_CLASSIF_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _INTENT_CLASSIF_USER_TMPL.render(
                    history=conversation.format_context(),
                    question=question,
                ),
            },
        ]

    @staticmethod
    def _build_context_answer_messages(question: str, conversation_context: str) -> list[dict]:
        """Format a data_question prompt — answer from conversation context, no SQL."""
        return [
            {"role": "system", "content": _CONTEXT_ANSWER_SYSTEM_TMPL.render()},
            {
                "role": "user",
                "content": _CONTEXT_ANSWER_USER_TMPL.render(
                    conversation_context=conversation_context,
                    question=question,
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

    @_tracer.start_as_current_span("llm.generate_sql")
    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        span = trace.get_current_span()
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
                    sql_to_format = sql.replace("\\n", "\n")
                    formatted_sql = sqlparse.format(
                        sql_to_format,
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
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, error)

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        logger.debug(
            "SQL generation completed",
            timing_ms=timing_ms,
            has_sql=sql is not None,
            has_error=error is not None,
            sql_preview=sql[:120],
        )
        return SQLGenerationOutput(
            sql=parsed.sql if parsed.answerable else None,
            answerable=parsed.answerable,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    @_tracer.start_as_current_span("llm.generate_answer")
    def generate_answer(
        self,
        question: str,
        sql: str | None,
        rows: list[dict[str, Any]],
        correction_hint: str = "",
        conversation_context: str = "",
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
                    question,
                    sql,
                    rows,
                    correction_hint=correction_hint,
                    rows_sample=self._answer_rows_sample,
                    conversation_context=conversation_context,
                ),
                temperature=self._answer_temperature,
                max_tokens=self._answer_max_tokens,
                reasoning_effort=self._answer_reasoning_effort,
            )
        except Exception as exc:
            logger.exception("Answer generation failed with an unexpected exception")
            error = str(exc)
            answer = f"Error generating answer: {error}"
            span = trace.get_current_span()
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, error)

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

    @_tracer.start_as_current_span("llm.judge_sql_analytics")
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
        span = trace.get_current_span()
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
            span.set_attribute("pipeline.judge_verdict", parsed.verdict)
            span.set_attribute("pipeline.judge_grade", parsed.grade)
            return SQLAnalyticsJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.exception("SQL analytics judge failed")
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return SQLAnalyticsJudgeOutput(verdict=False, grade="fail", issues=[], reason="", error=str(exc), llm_stats=llm_stats)

    @_tracer.start_as_current_span("llm.judge_answer_grounding")
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
        span = trace.get_current_span()
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
            span.set_attribute("pipeline.judge_verdict", parsed.verdict)
            span.set_attribute("pipeline.judge_grade", parsed.grade)
            return AnswerGroundingJudgeOutput(
                verdict=parsed.verdict,
                grade=parsed.grade,
                issues=parsed.issues,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.exception("Answer grounding judge failed")
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return AnswerGroundingJudgeOutput(verdict=False, grade="fail", issues=[], reason="", error=str(exc), llm_stats=llm_stats)

    @_tracer.start_as_current_span("llm.summarize_turns")
    def summarize_turns(self, turns: list) -> SummarizationOutput:
        """LLM summary of turns for context compression.

        Non-blocking: on any exception returns SummarizationOutput(summary="", error=...).
        WHY: empty on error retains original turns — safer than replacing them with wrong text.
        """
        span = trace.get_current_span()
        start = time.perf_counter()
        try:
            summary = self._chat(
                messages=self._build_summarization_messages(turns),
                temperature=self._answer_temperature,
                max_tokens=self._answer_max_tokens,
            )
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            logger.debug("Conversation summarization completed", turns_count=len(turns), timing_ms=(time.perf_counter() - start) * 1000)
            return SummarizationOutput(summary=summary, llm_stats=llm_stats)
        except Exception as exc:
            logger.exception("Conversation summarization failed")
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return SummarizationOutput(summary="", llm_stats=llm_stats, error=str(exc))

    @_tracer.start_as_current_span("llm.classify_intent")
    def classify_intent(self, question: str, conversation: Any) -> IntentClassificationOutput:
        """Classify whether question is follow_up, data_question, or new_query.

        Uses structured output (_IntentResponse). Non-blocking: on any exception
        returns follow_up (safe fallback).
        WHY: follow_up fallback is safer — over-injecting context is less harmful
        than missing context on a genuine follow-up question.
        """
        span = trace.get_current_span()
        start = time.perf_counter()
        try:
            text = self._chat(
                messages=self._build_intent_messages(question, conversation),
                temperature=0.0,
                max_tokens=self._intent_max_tokens,
                response_format=ChatFormatJSONSchemaConfig(
                    type="json_schema",
                    json_schema=ChatJSONSchemaConfig(
                        name="intent_classification",
                        schema_=_IntentResponse.model_json_schema(),
                        strict=True,
                    ),
                ),
            )
            parsed = _IntentResponse.model_validate_json(text)
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            logger.debug("Intent classification completed", intent=parsed.intent, timing_ms=(time.perf_counter() - start) * 1000)
            span.set_attribute("conv.intent", parsed.intent)
            return IntentClassificationOutput(
                intent=parsed.intent,
                reason=parsed.reason,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            logger.exception("Intent classification failed, falling back to follow_up")
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            return IntentClassificationOutput(
                intent="follow_up",
                reason="",
                llm_stats=llm_stats,
                error=str(exc),
            )

    @_tracer.start_as_current_span("llm.answer_from_context")
    def answer_from_context(self, question: str, conversation_context: str) -> AnswerGenerationOutput:
        """Answer directly from conversation context — no SQL cycle.

        Non-blocking: on any exception returns AnswerGenerationOutput with error set.
        WHY: data_question intent means the answer exists in prior results;
        a separate method with its own prompt keeps this path independent of
        the SQL-backed generate_answer flow.
        """
        span = trace.get_current_span()
        start = time.perf_counter()
        try:
            answer = self._chat(
                messages=self._build_context_answer_messages(question, conversation_context),
                temperature=self._answer_temperature,
                max_tokens=self._answer_max_tokens,
            )
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug("Context answer completed", timing_ms=timing_ms)
            return AnswerGenerationOutput(answer=answer, timing_ms=timing_ms, llm_stats=llm_stats)
        except Exception as exc:
            logger.exception("Context answer generation failed")
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            llm_stats = self.pop_stats()
            llm_stats["model"] = self.model
            timing_ms = (time.perf_counter() - start) * 1000
            return AnswerGenerationOutput(
                answer=f"Error generating answer: {exc}",
                timing_ms=timing_ms,
                llm_stats=llm_stats,
                error=str(exc),
            )

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
