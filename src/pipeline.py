from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import structlog

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.simplify import simplify

from src.config import PipelineConfig
from src.llm_client import OpenRouterLLMClient, build_default_llm_client
from src.schema_context import load_schema_context
from src.types import (
    PipelineOutput,
    ResultValidationOutput,
    SQLAnalyticsJudgeOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)

logger = structlog.get_logger()


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
        logger.debug("SQL validation started", sql_preview=(sql or "")[:120])
        start = time.perf_counter()

        def _invalid(error: str) -> SQLValidationOutput:
            timing_ms = (time.perf_counter() - start) * 1000
            logger.debug("SQL validation completed", is_valid=False, error=error, timing_ms=timing_ms)
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
        return SQLValidationOutput(
            is_valid=True,
            validated_sql=sql,
            error=None,
            timing_ms=timing_ms,
        )


class SQLiteExecutor:
    def __init__(self, db_path: str | Path, row_limit: int = 100) -> None:
        self.db_path = Path(db_path)
        # WHY: open once with mode=ro — enforces read-only at the OS level and
        # fails immediately if the file doesn't exist, rather than silently
        # creating an empty DB (sqlite3 default behaviour)
        self._conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._row_limit = row_limit

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(self, sql: str | None) -> SQLExecutionOutput:
        logger.debug("SQL execution started", sql_preview=(sql or "")[:120])
        start = time.perf_counter()
        error = None
        rows = []
        row_count = 0

        if sql is None:
            logger.debug("SQL execution skipped: no SQL provided")
            return SQLExecutionOutput(
                rows=[],
                row_count=0,
                timing_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )

        rows_truncated = False
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

        timing_ms = (time.perf_counter() - start) * 1000
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
        self.executor = SQLiteExecutor(self._config.db_path, row_limit=self._config.sql_row_limit)
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

    def _validate_and_execute(
        self,
        question: str,
        candidate_sql: str | None,
        sql_gen_output: SQLGenerationOutput,
        stats_sink: list[dict],
    ) -> tuple[SQLValidationOutput, str | None, SQLExecutionOutput, SQLAnalyticsJudgeOutput | None]:
        """Validate a SQL candidate, optionally judge it analytically, and execute it.

        WHY: the validate→judge→execute sequence is the same for both the initial
        SQL and each corrected SQL — extracting it removes the duplication.

        Appends judge llm_stats to stats_sink in-place (empty dict when disabled).
        Returns (validation_output, validated_sql_or_None, execution_output, judge_verdict_or_None).
        validated_sql_or_None is None when validation fails; caller breaks the
        correction loop on None.
        judge_verdict_or_None is None when judge is disabled or validation failed.
        """
        validation_output = SQLValidator.validate(candidate_sql, schema_context=self._schema_context)
        if not validation_output.is_valid:
            return validation_output, candidate_sql, SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0), None
        # WHY: judge is opt-in and non-blocking; stats appended to sink for aggregation
        judge_verdict: SQLAnalyticsJudgeOutput | None = None
        if self._config.sql_analytics_judge_enabled:
            judge_verdict = self.llm.judge_sql_analytics(question, candidate_sql, self._schema_context)
            sql_gen_output.intermediate_outputs.append(judge_verdict.model_dump())
            stats_sink.append(judge_verdict.llm_stats)
        execution_output = self.executor.run(candidate_sql)
        return validation_output, candidate_sql, execution_output, judge_verdict

    def run(self, question: str, request_id: str | None = None) -> PipelineOutput:
        """Public entry point. Owns context lifecycle; delegates logic to _run_impl."""
        # WHY: clear first so a previous run's context does not bleed into this one
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            pipeline_run_id=str(uuid.uuid4()),
            model=self.llm.model,
            **({"request_id": request_id} if request_id is not None else {}),
        )
        try:
            return self._run_impl(question, request_id)
        finally:
            structlog.contextvars.clear_contextvars()

    def _run_impl(self, question: str, request_id: str | None) -> PipelineOutput:
        start = time.perf_counter()

        # Stage 1: SQL Generation
        structlog.contextvars.bind_contextvars(stage="sql_generation")
        sql_gen_output = self.llm.generate_sql(question, self._schema_context)
        sql = sql_gen_output.sql

        # Stages 2 / 2b / 3: validate → judge (opt-in) → execute
        # WHY: _validate_and_execute encapsulates this shared sequence so the
        # unified correction loop reuses it without duplication.
        structlog.contextvars.bind_contextvars(stage="sql_validation")
        all_extra_stats: list[dict] = []
        validation_output, sql, execution_output, current_judge_verdict = self._validate_and_execute(
            question, sql_gen_output.sql, sql_gen_output, all_extra_stats
        )
        rows = execution_output.rows

        # Stage 3b: Result shape signals (informational — never blocks)
        structlog.contextvars.bind_contextvars(stage="result_validation")
        result_validation_output = ResultValidator.validate(execution_output, self._config.result_validation_enabled)

        # Unified SQL correction loop (opt-in per trigger type)
        # WHY: single loop handles all three correction triggers — each counter tracks
        # independently so one type of correction does not eat another's budget.
        # History accumulates across all iterations so the LLM sees every prior attempt.
        # Priority: execution error → analytics judge → result validation.
        execution_correction_attempts = 0
        analytics_correction_attempts = 0
        result_validation_correction_attempts = 0
        max_execution_retries = self._config.max_sql_correction_retries
        max_analytics_retries = self._config.max_sql_analytics_correction_retries
        max_result_validation_retries = self._config.max_result_validation_correction_retries
        correction_history: list[dict] = []

        while sql is not None:
            execution_should_correct = (
                self._config.sql_correction_enabled
                and (not validation_output.is_valid or execution_output.error is not None)
                and execution_correction_attempts < max_execution_retries
            )
            analytics_should_correct = (
                self._config.sql_analytics_judge_correction_enabled
                and current_judge_verdict is not None
                and not current_judge_verdict.verdict
                and not execution_output.error
                and analytics_correction_attempts < max_analytics_retries
            )
            result_should_correct = (
                self._config.result_validation_correction_enabled
                and validation_output.is_valid  # WHY: result signals are only meaningful after a successful execution
                and bool(result_validation_output.flags)
                and not execution_output.error
                and result_validation_correction_attempts < max_result_validation_retries
            )

            if not execution_should_correct and not analytics_should_correct and not result_should_correct:
                break

            # WHY priority: execution error blocks analytics/result checks entirely;
            # analytics correctness takes priority over result shape since fixing wrong
            # aggregation often resolves empty results too
            if execution_should_correct:
                execution_correction_attempts += 1
                # WHY: validation failure has no execution error — use the validation message instead
                current_hint = validation_output.error if not validation_output.is_valid else execution_output.error
                stage_name = "sql_correction"
            elif analytics_should_correct:
                analytics_correction_attempts += 1
                issues_hint = current_judge_verdict.reason
                current_hint = f"Analytics judge flagged the SQL as '{current_judge_verdict.grade}'. Issues: {issues_hint}"
                stage_name = "sql_analytics_correction"
            else:
                result_validation_correction_attempts += 1
                current_hint = f"Result validation flags: {result_validation_output.flags}"
                stage_name = "sql_result_validation_correction"

            structlog.contextvars.bind_contextvars(stage=stage_name)
            logger.debug(
                "SQL correction attempt triggered",
                trigger=stage_name,
                attempt=(
                    execution_correction_attempts if execution_should_correct
                    else analytics_correction_attempts if analytics_should_correct
                    else result_validation_correction_attempts
                ),
                hint=current_hint,
            )
            correction_history.append({"sql": sql, "hint": current_hint})
            history_lines = "\n".join(f"  {i + 1}. SQL: {h['sql']!r}\n     Reason: {h['hint']}" for i, h in enumerate(correction_history))
            full_hint = f"Previous failed attempts:\n{history_lines}"

            correction = self.llm.correct_sql(question, sql, full_hint, self._schema_context)
            sql_gen_output.intermediate_outputs.append(
                {
                    "stage": stage_name,
                    "corrected_sql": correction.sql,
                    "error": correction.error,
                }
            )
            all_extra_stats.append(correction.llm_stats)
            if correction.sql is None or correction.error:
                logger.warning(
                    "SQL correction returned no SQL, stopping correction loop",
                    trigger=stage_name,
                    error=correction.error,
                    attempt=(
                        execution_correction_attempts if execution_should_correct
                        else analytics_correction_attempts if analytics_should_correct
                        else result_validation_correction_attempts
                    ),
                )
                break

            corrected_validation, corrected_sql, corrected_execution, corrected_verdict = self._validate_and_execute(
                question, correction.sql, sql_gen_output, all_extra_stats
            )
            # WHY: always update sql + validation_output so correction_history reflects
            # the latest attempted SQL and its failure reason on subsequent iterations
            sql = corrected_sql
            validation_output = corrected_validation
            if not corrected_validation.is_valid:
                # Corrected SQL also invalid — loop retries if budget allows
                continue
            execution_output = corrected_execution
            rows = execution_output.rows
            current_judge_verdict = corrected_verdict
            result_validation_output = ResultValidator.validate(execution_output, self._config.result_validation_enabled)

        # WHY: warn only when budget ran out AND the problem still persists —
        # a correction that succeeded within budget is not a warning
        if (
            self._config.sql_correction_enabled
            and execution_correction_attempts >= max_execution_retries
            and (not validation_output.is_valid or execution_output.error is not None)
        ):
            logger.warning(
                "SQL correction budget exhausted, execution error persists",
                trigger="execution",
                attempts=execution_correction_attempts,
                max=max_execution_retries,
            )
        if (
            self._config.sql_analytics_judge_correction_enabled
            and analytics_correction_attempts >= max_analytics_retries
            and current_judge_verdict is not None
            and not current_judge_verdict.verdict
        ):
            logger.warning(
                "SQL correction budget exhausted, analytics judge still failing",
                trigger="analytics",
                attempts=analytics_correction_attempts,
                max=max_analytics_retries,
            )
        if (
            self._config.result_validation_correction_enabled
            and result_validation_correction_attempts >= max_result_validation_retries
            and bool(result_validation_output.flags)
        ):
            logger.warning(
                "SQL correction budget exhausted, result validation flags persist",
                trigger="result_validation",
                attempts=result_validation_correction_attempts,
                max=max_result_validation_retries,
            )

        # Stage 4: Answer Generation
        structlog.contextvars.bind_contextvars(stage="answer_generation")
        answer_output = self.llm.generate_answer(question, sql, rows)

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
                correction_hint = f"Previous answer was not grounded in the data. Issues: {issues_text}\nPrevious answer: {answer_output.answer}"
                corrected_answer_output = self.llm.generate_answer(question, sql, rows, correction_hint=correction_hint)
                all_extra_stats.append(corrected_answer_output.llm_stats)
                answer_output.intermediate_outputs.append({
                    "stage": "answer_grounding_correction",
                    "corrected_answer": corrected_answer_output.answer,
                    "error": corrected_answer_output.error,
                })
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
        if sql_gen_output.sql is None and sql_gen_output.error:
            # LLM could not produce SQL
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
            answer_output.llm_stats,
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
