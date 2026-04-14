from __future__ import annotations

import logging
import os
import sys

import structlog
from opentelemetry import trace


def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure structlog for the application.

    Called once from src/__init__.py on first import.
    Reads LOG_LEVEL and LOG_FORMAT from env if not passed explicitly.

    Args:
        level: DEBUG / INFO / WARNING / ERROR. Falls back to LOG_LEVEL env var,
               then "INFO".
        log_format: "json" or "pretty". Falls back to LOG_FORMAT env var,
                    then "json".
    """
    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    resolved_format = (log_format or os.environ.get("LOG_FORMAT", "json")).lower()

    # WHY: configure stdlib root logger so third-party libs that use standard
    # logging (e.g. httpx, tenacity) propagate through the same stream/level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, resolved_level, logging.INFO),
    )

    def _inject_otel_context(logger_arg, method: str, event_dict: dict) -> dict:
        """Bridge: inject trace_id + span_id from the active OTel span into structlog events.

        WHY: correlates structured logs with traces in Phoenix — a single
        trace_id links every log line to the span tree for that request.
        When no span is active (or OTel is disabled) the keys are omitted
        rather than injecting invalid zeros.
        """
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
        return event_dict

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        # WHY: module gives the logger-name equivalent for native structlog (no stdlib bridge);
        # filename + lineno give precise source attribution
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _inject_otel_context,
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if resolved_format == "pretty"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, resolved_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
