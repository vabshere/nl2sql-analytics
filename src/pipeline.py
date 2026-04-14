from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.simplify import simplify

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.config import PipelineConfig
from src.llm_client import OpenRouterLLMClient, build_default_llm_client
from src.schema_context import load_schema_context
from src.tracing import get_tracer
from src.types import (
    PipelineOutput,
    ResultValidationOutput,
    SQLAnalyticsJudgeOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)

logger = structlog.get_logger()
_tracer = get_tracer(__name__)


def _config_from_kwargs(
    config: PipelineConfig | None,
    db_path: str | Path | None,
    metadata_db_path: str | Path | None,
) -> PipelineConfig:
    """Resolve a PipelineConfig from explicit config or legacy path kwargs.

    WHY: db_path / metadata_db_path kwargs exist for backwards compatibility with
    callers that pre-date PipelineConfig (test_public.py, benchmark.py). When
    supplied they are merged as field overrides so the rest of the config (model,
    feature flags, timeouts, etc.) still comes from the environment.
    """
    if config is not None:
        return config
    overrides: dict = {}
    if db_path is not None:
        overrides["db_path"] = Path(db_path)
    if metadata_db_path is not None:
        overrides["metadata_db_path"] = Path(metadata_db_path)
    return PipelineConfig(**overrides) if overrides else PipelineConfig()


class SQLValidationError(Exception):
    pass


class SQLValidator:
    @classmethod
    def validate(
        cls,
        sql: str | None,
        schema_context: dict | None = None,
    ) -> SQLValidationOutput:
        with _tracer.start_as_current_span("pipeline.sql_validate") as _span:
            return cls._validate_impl(sql, schema_context, _span)

    @classmethod
    def _validate_impl(
        cls,
        sql: str | None,
        schema_context: dict | None,
        _span: trace.Span,
    ) -> SQLValidationOutput:
        logger.debug("SQL validation started", sql_preview=(sql or "")[:120])
        start = time.perf_counter()

        def _invalid(error: str) -> SQLValidationOutput:
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug("SQL validation completed", is_valid=False, error=error, timing_ms=timing_ms)
            _span.set_attribute("sql.is_valid", False)
            _span.set_attribute("sql.error", error)
            _span.set_status(StatusCode.ERROR, error)
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=error,
                timing_ms=timing_ms,
            )

        # Layer 0: None / empty guard
        if not sql or not sql.strip():
            return _invalid("No SQL provided")

        # Layer 1: parse + statement-type check via sqlglot
        # WHY: sqlglot raises ParseError on invalid syntax AND gives us a typed
        # AST, letting us check statement type and structure in one pass without
        # a separate DB round-trip
        try:
            parsed = sqlglot.parse(sql, dialect="sqlite", error_level=ErrorLevel.RAISE)
        except sqlglot.errors.ParseError as exc:
            return _invalid(f"SQL syntax error: {exc}")

        if len(parsed) > 1:
            return _invalid("Multi-statement SQL is not allowed.")

        if not isinstance(parsed[0], exp.Select):
            stmt_type = type(parsed[0]).__name__.upper()
            return _invalid(f"SQL statement type '{stmt_type}' is not allowed; only SELECT queries are permitted.")

        stmt = parsed[0]

        # Layer 2: schema + semantic validation
        if schema_context:
            known_tables = {t.lower() for t in schema_context.get("tables", set())}
            referenced_tables = {t.name.lower() for t in stmt.find_all(exp.Table)}

            # 2a. Must reference at least one real table
            if not referenced_tables:
                return _invalid("Query must reference at least one table.")

            # 2b. All referenced tables must exist in the schema
            unknown = referenced_tables - known_tables
            if unknown:
                return _invalid(f"References unknown table(s): {sorted(unknown)}")

            # 2c. Reject always-true WHERE clause (1=1, OR 1=1, WHERE TRUE, etc.)
            # WHY: simplify() removes the WHERE entirely when the condition is a
            # tautology; checking original-has-WHERE vs simplified-has-none
            # catches non-obvious forms like 1+0=1 or 'x' LIKE 'x' too
            if stmt.find(exp.Where):
                try:
                    simplified = simplify(stmt)
                except Exception as exc:
                    return _invalid(f"WHERE clause simplification failed: {exc}")
                if not simplified.find(exp.Where):
                    return _invalid("WHERE clause is always true.")

        timing_ms = (time.perf_counter() - start) * 1000
        logger.debug("SQL validation completed", is_valid=True, timing_ms=timing_ms)
        _span.set_attribute("sql.is_valid", True)
        return SQLValidationOutput(
            is_valid=True,
            validated_sql=sql,
            error=None,
            timing_ms=timing_ms,
        )


class SQLiteExecutor:
    def __init__(self, db_path: str | Path, row_limit: int = 100, config: PipelineConfig | None = None) -> None:
        self.db_path = Path(db_path)
        # WHY: open once with mode=ro — enforces read-only at the OS level and
        # fails immediately if the file doesn't exist, rather than silently
        # creating an empty DB (sqlite3 default behaviour)
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._row_limit = row_limit
        # WHY: stored so run() can read tracing flags (otlp_include_sql) without
        # callers passing config on every call; None is safe — defaults to True
        self._config = config

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @_tracer.start_as_current_span("db.execute")
    def run(self, sql: str | None) -> SQLExecutionOutput:
        span = trace.get_current_span()
        span.set_attribute("db.system", "sqlite")
        logger.debug("SQL execution started", sql_preview=(sql or "")[:120])
        start = time.perf_counter()
        error = None
        rows = []
        row_count = 0
        rows_truncated = False

        if sql is None:
            logger.debug("SQL execution skipped: no SQL provided")
            return SQLExecutionOutput(
                rows=[],
                row_count=0,
                timing_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )

        # WHY: otlp_include_sql is checked at config level — the executor stores
        # the config reference so it can respect the flag without needing callers
        # to pass it on every call
        if getattr(self, "_config", None) and getattr(self._config, "otlp_include_sql", True):
            span.set_attribute("db.statement", sql)

        try:
            cur = self._conn.cursor()
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(self._row_limit)]
            row_count = len(rows)
            # WHY: probe for a (row_limit + 1)th row only when the batch is full —
            # if fewer than row_limit rows came back we know there was no truncation
            # and the extra fetchone() round-trip is unnecessary
            if row_count == self._row_limit:
                rows_truncated = cur.fetchone() is not None
        except Exception as exc:
            logger.exception("SQL execution failed with an unexpected exception")
            error = str(exc)
            rows = []
            row_count = 0
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, error)

        timing_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("db.row_count", row_count)
        span.set_attribute("db.rows_truncated", rows_truncated)
        logger.debug(
            "SQL execution completed",
            row_count=row_count,
            rows_truncated=rows_truncated,
            timing_ms=timing_ms,
            has_error=error is not None,
        )
        return SQLExecutionOutput(
            rows=rows,
            row_count=row_count,
            timing_ms=timing_ms,
            error=error,
            rows_truncated=rows_truncated,
        )


class ResultValidator:
    @classmethod
    def validate(cls, execution_output: SQLExecutionOutput, result_validation_enabled: bool = True) -> ResultValidationOutput:
        """Collect post-execution result shape signals.

        WHY: these are monitoring/observability flags only — never block the pipeline.
        The empty-rows answer path is already handled in generate_answer(); this
        adds structured metadata for logging and potential future correction loops.
        """
        logger.debug("Result validation started", row_count=execution_output.row_count, enabled=result_validation_enabled)
        start = time.perf_counter()

        # WHY: caller passes the flag from config so this method has no env dependency
        if not result_validation_enabled:
            logger.debug("Result validation skipped: disabled")
            return ResultValidationOutput(flags=[], timing_ms=0.0)

        # WHY: if execution itself errored, rows may be empty/partial for reasons
        # unrelated to query semantics — skip checks to avoid false signals
        if execution_output.error is not None:
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug("Result validation skipped: execution error", timing_ms=timing_ms)
            return ResultValidationOutput(
                flags=[],
                timing_ms=timing_ms,
            )

        flags: list[str] = []
        rows = execution_output.rows

        # Flag 1: empty result (SQL ran successfully but returned no rows)
        if not rows:
            flags.append("empty_result")
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug("Result validation completed", flags=flags, timing_ms=timing_ms)
            return ResultValidationOutput(
                flags=flags,
                timing_ms=timing_ms,
            )

        # Flag 2: result was truncated at 100 rows (more rows exist in the DB)
        if execution_output.rows_truncated:
            flags.append("rows_truncated")

        # Flag 3 & 4: column-level signals (only meaningful when rows exist)
        cols = rows[0].keys()
        for col in cols:
            values = [row[col] for row in rows]

            # all_null_column: every value is None
            if all(v is None for v in values):
                flags.append(f"all_null_column:{col}")
                continue

            # all_zero_column: every numeric value is 0 / 0.0
            # WHY: only check numeric types — text "0" is not a zero signal
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values and len(numeric_values) == len(values) and all(v == 0 for v in numeric_values):
                flags.append(f"all_zero_column:{col}")

        timing_ms = (time.perf_counter() - start) * 1000
        logger.debug("Result validation completed", flags=flags, timing_ms=timing_ms)
        return ResultValidationOutput(
            flags=flags,
            timing_ms=timing_ms,
        )


@dataclass
class _CycleOutput:
    """Return value of _run_sql_cycle — one full generate→validate→judge→execute→result-validate cycle.

    WHY: a named type is clearer than a 6-element tuple and makes callers
    resilient to future field additions.

    Caller contract:
      sql is None         → generation produced no SQL; caller must stop (break the loop).
      sql is non-None AND validation.is_valid is False
                          → SQL was generated but is syntactically/semantically invalid;
                            caller should continue so correction_history records the attempt.
      sql is non-None AND validation.is_valid is True AND judge_verdict is not None AND not judge_verdict.verdict
                          → analytics judge failed and execution was skipped;
                            caller should continue to trigger analytics correction.
      sql is non-None AND validation.is_valid is True AND execution has no error
                          → happy path; rows, result_validation, and answer generation proceed.

    result_validation.flags is always empty when execution was intentionally skipped
    (analytics judge failing with correction enabled) — avoids false empty_result signals.
    """

    # initial: newly created; correction: same object passed in, mutated
    sql_gen_output: SQLGenerationOutput
    sql: str | None  # None only when generation failed; non-None even when invalid
    validation: SQLValidationOutput
    execution: SQLExecutionOutput  # empty stub when execution was skipped
    # None when judge disabled or generation/validation failed
    judge_verdict: SQLAnalyticsJudgeOutput | None
    # empty flags when execution was skipped
    result_validation: ResultValidationOutput


class AnalyticsPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        llm_client: OpenRouterLLMClient | None = None,
        *,
        db_path: str | Path | None = None,
        metadata_db_path: str | Path | None = None,
    ) -> None:
        # WHY: llm_client is kept as a separate param for test stub injection —
        # it is dependency injection, not configuration
        self._config = _config_from_kwargs(config, db_path, metadata_db_path)
        self.llm = llm_client or build_default_llm_client(self._config)
        self.executor = SQLiteExecutor(self._config.db_path, row_limit=self._config.sql_row_limit, config=self._config)
        # WHY: build once at init — schema is static, no per-request DB roundtrip
        try:
            self._schema_context = load_schema_context(
                self._config.db_path,
                self._config.metadata_db_path,
                self._config.table_name,
                self._config.schema_include_description,
            )
        except sqlite3.OperationalError:
            logger.exception(
                "Could not introspect schema — generate_sql will receive empty context",
                db_path=str(self._config.db_path),
            )
            self._schema_context = {}
        except Exception:
            logger.exception(
                "Unexpected error building schema context — generate_sql will receive empty context",
                db_path=str(self._config.db_path),
            )
            self._schema_context = {}

    def close(self) -> None:
        self.executor.close()

    def __enter__(self) -> "AnalyticsPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # WHY: best-effort safety net for callers that don't use the context
        # manager — GC timing is not guaranteed so this is not a substitute for
        # explicit close() or the with-statement
        try:
            self.executor.close()
        except Exception:
            pass

    def _should_skip_execution(self, judge_verdict: SQLAnalyticsJudgeOutput | None) -> bool:
        """Return True when the analytics judge failed and correction is enabled.

        WHY: executing analytically wrong SQL wastes a DB round-trip and would push
        incorrect rows into execution_output before the next correction replaces the SQL.
        """
        return judge_verdict is not None and not judge_verdict.verdict and self._config.sql_analytics_judge_correction_enabled

    def _should_correct_execution(
        self,
        validation_output: SQLValidationOutput,
        execution_output: SQLExecutionOutput,
        attempts: int,
    ) -> bool:
        """Return True when a validation failure or DB execution error warrants SQL correction.

        WHY: covers two distinct failure modes under one flag — static validation
        rejection (SQL never ran) and runtime DB error (SQL ran but failed).
        In both cases the fix is the same: regenerate SQL with the error as hint.
        """
        return (
            self._config.sql_correction_enabled
            and (not validation_output.is_valid or execution_output.error is not None)
            and attempts < self._config.max_sql_correction_retries
        )

    def _should_correct_analytics(
        self,
        judge_verdict: SQLAnalyticsJudgeOutput | None,
        attempts: int,
    ) -> bool:
        """Return True when the analytics judge failed and correction is enabled.

        WHY: called only after the execution branch was evaluated — priority guarantees
        that an execution error did not also fire, so no re-check is needed here.
        """
        return (
            self._config.sql_analytics_judge_correction_enabled
            and judge_verdict is not None
            and not judge_verdict.verdict
            and attempts < self._config.max_sql_analytics_correction_retries
        )

    def _should_correct_result_validation(
        self,
        validation_output: SQLValidationOutput,
        result_validation_output: ResultValidationOutput,
        attempts: int,
    ) -> bool:
        """Return True when result validation flags warrant SQL correction.

        WHY: result signals are only meaningful after a clean execution — the
        validation_output.is_valid guard ensures execution actually ran.
        Called only after execution and analytics branches were evaluated (priority).
        """
        return (
            self._config.result_validation_correction_enabled
            and validation_output.is_valid
            and bool(result_validation_output.flags)
            and attempts < self._config.max_result_validation_correction_retries
        )

    @_tracer.start_as_current_span("pipeline.sql_cycle")
    def _run_sql_cycle(
        self,
        question: str,
        stats_sink: list[dict],
        sql_gen_output: SQLGenerationOutput | None = None,
        correction_hint: str | None = None,
        stage_name: str = "sql_correction",
        cycle_attempt: int = 0,
        correction_trigger: str = "",
        conversation_context: str = "",
    ) -> _CycleOutput:
        """Own the full per-iteration SQL cycle: generate → validate → judge → execute → result-validate.

        WHY: consolidating all five sub-stages here removes duplication between the
        initial call in _run_impl and each correction iteration, and ensures result
        validation never runs on a skipped execution (avoiding false empty_result flags).

        stats_sink: judge and correction llm_stats are appended in-place.
        sql_gen_output: must be provided for correction calls (correction_hint is not None).
        correction_hint: when None the initial path runs; when set the correction path runs.
        """
        span = trace.get_current_span()
        span.set_attribute("pipeline.cycle_attempt", cycle_attempt)
        span.set_attribute("pipeline.correction_trigger", correction_trigger)

        _empty_execution = SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0)
        _empty_result = ResultValidationOutput(flags=[], timing_ms=0.0)

        context = self._schema_context if correction_hint is None else {**self._schema_context, "correction_hint": correction_hint}
        if conversation_context:
            # WHY: only inject when non-empty — correction cycles reuse this method
            # and must not lose the context on the initial call's value
            context = {**context, "conversation_context": conversation_context}
        gen_output = self.llm.generate_sql(question, context)

        # Path-specific bookkeeping only — common pipeline logic follows
        if correction_hint is None:
            # Initial path: gen_output becomes the canonical sql_gen_output for this run
            sql_gen_output = gen_output
        else:
            # Correction path: sql_gen_output must be provided by the caller
            assert sql_gen_output is not None, "sql_gen_output must be provided for correction calls"
            sql_gen_output.intermediate_outputs.append(
                {
                    "stage": stage_name,
                    "corrected_sql": gen_output.sql,
                    "error": gen_output.error,
                }
            )
            stats_sink.append(gen_output.llm_stats)

        # WHY: error (LLM call failed) and unanswerable (answerable=False) are distinct —
        # error is logged as a warning; unanswerable is a valid LLM response, not an error
        if gen_output.error:
            logger.warning("SQL generation failed", error=gen_output.error)
            return _CycleOutput(
                sql_gen_output=sql_gen_output,
                sql=None,
                validation=SQLValidationOutput(is_valid=False, validated_sql=None, error=gen_output.error, timing_ms=0.0),
                execution=_empty_execution,
                judge_verdict=None,
                result_validation=_empty_result,
            )

        if gen_output.sql is None:
            # answerable=False: LLM determined the question cannot be answered from the schema
            return _CycleOutput(
                sql_gen_output=sql_gen_output,
                sql=None,
                validation=SQLValidationOutput(is_valid=False, validated_sql=None, error="Question is unanswerable", timing_ms=0.0),
                execution=_empty_execution,
                judge_verdict=None,
                result_validation=_empty_result,
            )

        candidate_sql = gen_output.sql

        validation_output = SQLValidator.validate(candidate_sql, schema_context=self._schema_context)
        if not validation_output.is_valid:
            # WHY: return candidate_sql (never None here) so correction_history records
            # this attempt's SQL on the next iteration
            return _CycleOutput(
                sql_gen_output=sql_gen_output,
                sql=candidate_sql,
                validation=validation_output,
                execution=_empty_execution,
                judge_verdict=None,
                result_validation=_empty_result,
            )

        # WHY: judge is opt-in and non-blocking; stats appended to sink for aggregation
        judge_verdict: SQLAnalyticsJudgeOutput | None = None
        if self._config.sql_analytics_judge_enabled:
            judge_verdict = self.llm.judge_sql_analytics(question, candidate_sql, self._schema_context)
            sql_gen_output.intermediate_outputs.append(judge_verdict.model_dump())
            stats_sink.append(judge_verdict.llm_stats)

        if self._should_skip_execution(judge_verdict):
            return _CycleOutput(
                sql_gen_output=sql_gen_output,
                sql=candidate_sql,
                validation=validation_output,
                execution=_empty_execution,
                judge_verdict=judge_verdict,
                result_validation=_empty_result,
            )

        execution_output = self.executor.run(candidate_sql)
        if execution_output.error is not None:
            # WHY: result signals are only meaningful after a clean execution —
            # rows may be empty or partial for reasons unrelated to query semantics
            return _CycleOutput(
                sql_gen_output=sql_gen_output,
                sql=candidate_sql,
                validation=validation_output,
                execution=execution_output,
                judge_verdict=judge_verdict,
                result_validation=_empty_result,
            )
        result_validation_output = ResultValidator.validate(execution_output, self._config.result_validation_enabled)
        return _CycleOutput(
            sql_gen_output=sql_gen_output,
            sql=candidate_sql,
            validation=validation_output,
            execution=execution_output,
            judge_verdict=judge_verdict,
            result_validation=result_validation_output,
        )

    @_tracer.start_as_current_span("pipeline.run")
    def run(
        self,
        question: str,
        request_id: str | None = None,
        conversation_context: str = "",
    ) -> PipelineOutput:
        """Public entry point. Owns context lifecycle; delegates logic to _run_impl."""
        span = trace.get_current_span()
        pipeline_run_id = str(uuid.uuid4())
        span.set_attribute("pipeline.run_id", pipeline_run_id)
        span.set_attribute("pipeline.request_id", request_id or "")
        span.set_attribute("pipeline.model", self.llm.model)

        # WHY: clear first so a previous run's context does not bleed into this one
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            pipeline_run_id=pipeline_run_id,
            model=self.llm.model,
            **({"request_id": request_id} if request_id is not None else {}),
        )
        try:
            result = self._run_impl(question, request_id, conversation_context)
            span.set_attribute("pipeline.status", result.status)
            span.set_attribute("pipeline.total_ms", result.timings.get("total_ms", 0.0))
            span.set_attribute("pipeline.llm_calls", int(result.total_llm_stats.get("llm_calls", 0)))
            span.set_attribute("pipeline.total_tokens", int(result.total_llm_stats.get("total_tokens", 0)))
            # WHY: any non-success outcome (unanswerable, invalid_sql, error) marks
            # the root span ERROR so Phoenix dashboards surface pipeline failures,
            # not just hard exceptions
            if result.status != "success":
                span.set_status(StatusCode.ERROR, result.status)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise
        finally:
            structlog.contextvars.clear_contextvars()

    def _run_impl(self, question: str, request_id: str | None, conversation_context: str = "") -> PipelineOutput:
        start = time.perf_counter()

        # Stages 1–3b: generate → validate → judge (opt-in) → execute → result-validate
        # WHY: _run_sql_cycle owns the full cycle; _run_impl only determines
        # the trigger and correction hint for each iteration
        structlog.contextvars.bind_contextvars(stage="sql_generation")
        all_extra_stats: list[dict] = []
        cycle = self._run_sql_cycle(question, all_extra_stats, cycle_attempt=0, conversation_context=conversation_context)
        sql_gen_output = cycle.sql_gen_output
        sql = cycle.sql
        validation_output = cycle.validation
        execution_output = cycle.execution
        current_judge_verdict = cycle.judge_verdict
        result_validation_output = cycle.result_validation
        rows = execution_output.rows

        # Unified SQL correction loop (opt-in per trigger type)
        # WHY: single loop handles all three correction triggers — each counter tracks
        # independently so one type of correction does not eat another's budget.
        # History accumulates across all iterations so the LLM sees every prior attempt.
        # Priority: execution error → analytics judge → result validation.
        execution_correction_attempts = 0
        analytics_correction_attempts = 0
        result_validation_correction_attempts = 0
        correction_history: list[dict] = []

        while sql is not None:
            # WHY: evaluate triggers via helpers in priority order — only one fires
            # per iteration, preventing spurious lower-priority corrections
            if self._should_correct_execution(validation_output, execution_output, execution_correction_attempts):
                execution_correction_attempts += 1
                current_attempt = execution_correction_attempts
                # WHY: validation failure has no execution error — use validation message instead
                current_hint = validation_output.error if not validation_output.is_valid else execution_output.error
                stage_name = "sql_correction"
            elif self._should_correct_analytics(current_judge_verdict, analytics_correction_attempts):
                analytics_correction_attempts += 1
                current_attempt = analytics_correction_attempts
                current_hint = f"Analytics judge flagged the SQL as '{current_judge_verdict.grade}'. Issues: {current_judge_verdict.reason}"
                stage_name = "sql_analytics_correction"
            elif self._should_correct_result_validation(validation_output, result_validation_output, result_validation_correction_attempts):
                result_validation_correction_attempts += 1
                current_attempt = result_validation_correction_attempts
                current_hint = f"Result validation flags: {result_validation_output.flags}"
                stage_name = "sql_result_validation_correction"
            else:
                logger.debug(
                    "SQL correction loop: no trigger matched, stopping",
                    execution_attempts=execution_correction_attempts,
                    analytics_attempts=analytics_correction_attempts,
                    result_validation_attempts=result_validation_correction_attempts,
                )
                break

            structlog.contextvars.bind_contextvars(stage=stage_name)
            logger.debug(
                "SQL correction attempt triggered",
                trigger=stage_name,
                attempt=current_attempt,
                hint=current_hint,
            )
            correction_history.append({"sql": sql, "hint": current_hint})
            history_lines = "\n".join(f"  {i + 1}. SQL: {h['sql']!r}\n     Reason: {h['hint']}" for i, h in enumerate(correction_history))
            full_hint = f"Previous failed attempts:\n{history_lines}"

            cycle = self._run_sql_cycle(
                question,
                all_extra_stats,
                sql_gen_output=sql_gen_output,
                correction_hint=full_hint,
                stage_name=stage_name,
                cycle_attempt=max(execution_correction_attempts, analytics_correction_attempts, result_validation_correction_attempts),
                correction_trigger=stage_name,
            )
            # WHY: always update sql + validation_output so correction_history reflects
            # the latest attempted SQL and its failure reason on subsequent iterations
            if cycle.sql is None:
                break  # generation failed or unanswerable — _run_sql_cycle already logged
            sql = cycle.sql
            validation_output = cycle.validation
            if not cycle.validation.is_valid:
                continue  # corrected SQL also invalid — loop retries if budget allows
            execution_output = cycle.execution
            rows = execution_output.rows
            current_judge_verdict = cycle.judge_verdict
            result_validation_output = cycle.result_validation

        # WHY: warn only when budget ran out AND the problem still persists —
        # a correction that succeeded within budget is not a warning
        if (
            self._config.sql_correction_enabled
            and execution_correction_attempts >= self._config.max_sql_correction_retries
            and (not validation_output.is_valid or execution_output.error is not None)
        ):
            logger.warning(
                "SQL correction budget exhausted, execution error persists",
                trigger="execution",
                attempts=execution_correction_attempts,
                max=self._config.max_sql_correction_retries,
            )
        if (
            self._config.sql_analytics_judge_correction_enabled
            and analytics_correction_attempts >= self._config.max_sql_analytics_correction_retries
            and current_judge_verdict is not None
            and not current_judge_verdict.verdict
        ):
            logger.warning(
                "SQL correction budget exhausted, analytics judge still failing",
                trigger="analytics",
                attempts=analytics_correction_attempts,
                max=self._config.max_sql_analytics_correction_retries,
            )
        if (
            self._config.result_validation_correction_enabled
            and result_validation_correction_attempts >= self._config.max_result_validation_correction_retries
            and bool(result_validation_output.flags)
        ):
            logger.warning(
                "SQL correction budget exhausted, result validation flags persist",
                trigger="result_validation",
                attempts=result_validation_correction_attempts,
                max=self._config.max_result_validation_correction_retries,
            )

        # Stage 4: Answer Generation
        structlog.contextvars.bind_contextvars(stage="answer_generation")
        answer_output = self.llm.generate_answer(question, sql, rows, conversation_context=conversation_context)
        # WHY: save before grounding correction may replace answer_output — the
        # initial stats must reach _all_stats even when corrections occur
        initial_answer_llm_stats = answer_output.llm_stats

        # Stage 4b: Answer grounding judge + correction loop (opt-in, non-blocking)
        # WHY: skip when sql or rows are absent — the fallback answer paths have
        # no substantive data to verify against, making grounding checks meaningless
        answer_judge_llm_stats: dict = {}
        if self._config.answer_grounding_judge_enabled and sql is not None and rows:
            grounding_verdict = self.llm.judge_answer_grounding(question, sql, rows, answer_output.answer)
            answer_judge_llm_stats = grounding_verdict.llm_stats
            answer_output.intermediate_outputs.append(grounding_verdict.model_dump())

            answer_grounding_correction_attempts = 0
            max_answer_grounding_retries = self._config.max_answer_grounding_correction_retries
            # WHY: re-generate answer with same sql+rows — grounding failures are answer
            # phrasing issues, not data retrieval issues; no SQL re-generation needed
            while (
                self._config.answer_grounding_judge_correction_enabled
                and not grounding_verdict.verdict
                and answer_grounding_correction_attempts < max_answer_grounding_retries
            ):
                answer_grounding_correction_attempts += 1
                issues_text = "; ".join(grounding_verdict.issues) if grounding_verdict.issues else grounding_verdict.reason
                correction_hint = (
                    f"Previous answer was not grounded in the data. Issues: {issues_text}\nPrevious answer: {answer_output.answer}"
                )
                corrected_answer_output = self.llm.generate_answer(question, sql, rows, correction_hint=correction_hint)
                all_extra_stats.append(corrected_answer_output.llm_stats)
                answer_output.intermediate_outputs.append(
                    {
                        "stage": "answer_grounding_correction",
                        "corrected_answer": corrected_answer_output.answer,
                        "error": corrected_answer_output.error,
                    }
                )
                if corrected_answer_output.error:
                    logger.warning(
                        "Answer grounding correction failed, stopping correction loop",
                        error=corrected_answer_output.error,
                        attempt=answer_grounding_correction_attempts,
                    )
                    break
                # WHY: carry over accumulated intermediate_outputs before replacing —
                # corrected_answer_output starts with an empty list
                corrected_answer_output.intermediate_outputs = answer_output.intermediate_outputs
                answer_output = corrected_answer_output
                grounding_verdict = self.llm.judge_answer_grounding(question, sql, rows, answer_output.answer)
                all_extra_stats.append(grounding_verdict.llm_stats)
                answer_output.intermediate_outputs.append(grounding_verdict.model_dump())

        # Determine status — check every stage error in pipeline order
        # WHY: explicitly listing answer_output.error as the final error check ensures
        # every stage's failure is represented, and the ordering preserves existing
        # precedence (sql failures > execution failures > answer failures)
        status = "success"
        if sql_gen_output.sql is None and (sql_gen_output.error or not sql_gen_output.answerable):
            # LLM either errored or explicitly said the question is unanswerable
            status = "unanswerable"
        elif not validation_output.is_valid:
            status = "invalid_sql"
        elif execution_output.error:
            status = "error"
        elif sql is None:
            # Validation cleared but sql was reset (e.g. always-true WHERE → invalid)
            status = "unanswerable"
        elif answer_output.error:
            # Answer generation failed after successful SQL execution
            status = "error"

        # Build timings aggregate
        timings = {
            "sql_generation_ms": sql_gen_output.timing_ms,
            "sql_validation_ms": validation_output.timing_ms,
            "sql_execution_ms": execution_output.timing_ms,
            "answer_generation_ms": answer_output.timing_ms,
            "total_ms": (time.perf_counter() - start) * 1000,
        }

        # Build total LLM stats
        _all_stats = [
            sql_gen_output.llm_stats,
            initial_answer_llm_stats,
            answer_judge_llm_stats,
            *all_extra_stats,
        ]
        total_llm_stats = {
            key: sum(s.get(key, 0) for s in _all_stats) for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        total_llm_stats["model"] = sql_gen_output.llm_stats.get("model", "unknown")

        logger.info(
            "Pipeline run completed",
            status=status,
            sql_generation_ms=timings["sql_generation_ms"],
            sql_validation_ms=timings["sql_validation_ms"],
            sql_execution_ms=timings["sql_execution_ms"],
            answer_generation_ms=timings["answer_generation_ms"],
            total_ms=timings["total_ms"],
            llm_calls=total_llm_stats["llm_calls"],
            total_tokens=total_llm_stats["total_tokens"],
        )

        return PipelineOutput(
            status=status,
            question=question,
            request_id=request_id,
            sql_generation=sql_gen_output,
            sql_validation=validation_output,
            sql_execution=execution_output,
            answer_generation=answer_output,
            sql=sql,
            rows=rows,
            answer=answer_output.answer,
            timings=timings,
            total_llm_stats=total_llm_stats,
            result_validation=result_validation_output,
        )
