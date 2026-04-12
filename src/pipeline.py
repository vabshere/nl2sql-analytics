from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.simplify import simplify

from src.llm_client import OpenRouterLLMClient, build_default_llm_client
from src.schema_context import load_schema_context
from src.types import (
    PipelineOutput,
    SQLExecutionOutput,
    SQLValidationOutput,
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "data" / "gaming_mental_health.sqlite"
DEFAULT_METADATA_DB_PATH = BASE_DIR / "data" / "schema_metadata.sqlite"


class SQLValidationError(Exception):
    pass


class SQLValidator:
    @classmethod
    def validate(
        cls,
        sql: str | None,
        schema_context: dict | None = None,
    ) -> SQLValidationOutput:
        start = time.perf_counter()

        def _invalid(error: str) -> SQLValidationOutput:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=error,
                timing_ms=(time.perf_counter() - start) * 1000,
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
            if stmt.find(exp.Where) and not simplify(stmt).find(exp.Where):
                return _invalid("WHERE clause is always true.")

        return SQLValidationOutput(
            is_valid=True,
            validated_sql=sql,
            error=None,
            timing_ms=(time.perf_counter() - start) * 1000,
        )


class SQLiteExecutor:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def run(self, sql: str | None) -> SQLExecutionOutput:
        start = time.perf_counter()
        error = None
        rows = []
        row_count = 0

        if sql is None:
            return SQLExecutionOutput(
                rows=[],
                row_count=0,
                timing_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchmany(100)]
                row_count = len(rows)
        except Exception as exc:
            error = str(exc)
            rows = []
            row_count = 0

        return SQLExecutionOutput(
            rows=rows,
            row_count=row_count,
            timing_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )


class AnalyticsPipeline:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        llm_client: OpenRouterLLMClient | None = None,
        metadata_db_path: str | Path = DEFAULT_METADATA_DB_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.llm = llm_client or build_default_llm_client()
        self.executor = SQLiteExecutor(self.db_path)
        # WHY: build once at init — schema is static, no per-request DB roundtrip
        try:
            self._schema_context = load_schema_context(self.db_path, Path(metadata_db_path))
        except sqlite3.OperationalError as exc:
            logger.warning(
                "Could not introspect schema from %s (%s) — generate_sql will receive empty context.",
                self.db_path,
                exc,
            )
            self._schema_context = {}
        except Exception as exc:
            logger.error(
                "Unexpected error building schema context: %s — generate_sql will receive empty context.",
                exc,
            )
            self._schema_context = {}

    def run(self, question: str, request_id: str | None = None) -> PipelineOutput:
        start = time.perf_counter()

        # Stage 1: SQL Generation
        sql_gen_output = self.llm.generate_sql(question, self._schema_context)
        sql = sql_gen_output.sql

        # Stage 2: SQL Validation
        validation_output = SQLValidator.validate(sql, schema_context=self._schema_context)
        if not validation_output.is_valid:
            sql = None

        # Stage 3: SQL Execution
        execution_output = self.executor.run(sql)
        rows = execution_output.rows

        # Stage 4: Answer Generation
        answer_output = self.llm.generate_answer(question, sql, rows)

        # Determine status
        status = "success"
        if sql_gen_output.sql is None and sql_gen_output.error:
            status = "unanswerable"
        elif not validation_output.is_valid:
            status = "invalid_sql"
        elif execution_output.error:
            status = "error"
        elif sql is None:
            status = "unanswerable"

        # Build timings aggregate
        timings = {
            "sql_generation_ms": sql_gen_output.timing_ms,
            "sql_validation_ms": validation_output.timing_ms,
            "sql_execution_ms": execution_output.timing_ms,
            "answer_generation_ms": answer_output.timing_ms,
            "total_ms": (time.perf_counter() - start) * 1000,
        }

        # Build total LLM stats
        total_llm_stats = {
            "llm_calls": sql_gen_output.llm_stats.get("llm_calls", 0) + answer_output.llm_stats.get("llm_calls", 0),
            "prompt_tokens": sql_gen_output.llm_stats.get("prompt_tokens", 0) + answer_output.llm_stats.get("prompt_tokens", 0),
            "completion_tokens": sql_gen_output.llm_stats.get("completion_tokens", 0) + answer_output.llm_stats.get("completion_tokens", 0),
            "total_tokens": sql_gen_output.llm_stats.get("total_tokens", 0) + answer_output.llm_stats.get("total_tokens", 0),
            "model": sql_gen_output.llm_stats.get("model", "unknown"),
        }

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
        )
