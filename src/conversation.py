"""Conversation management for multi-turn analytics sessions.

Provides three public types:

  ConversationTurn     — immutable record of one completed pipeline turn
  Conversation         — pure state: turn storage, token counting, context formatting
  ConversationSession  — stateful wrapper around AnalyticsPipeline; owns one Conversation
                         per session; handles summarization + intent classification before
                         each pipeline call

Design invariants:
  - Conversation has NO dependency on the LLM or pipeline — it is purely testable.
  - ConversationSession owns all LLM conversation operations and delegates SQL/answer
    work to the shared AnalyticsPipeline instance.
  - AnalyticsPipeline sees only a pre-formatted conversation_context string — it never
    receives a Conversation object.
  - The feature is gated by PipelineConfig.conversation_history_enabled; when False,
    ConversationSession.run() is a zero-overhead passthrough.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
import tiktoken
from opentelemetry import trace

from src.config import PipelineConfig
from src.tracing import get_tracer
from src.types import (
    AnswerGenerationOutput,
    IntentClassificationOutput,
    PipelineOutput,
    ResultValidationOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
    SummarizationOutput,
)

if TYPE_CHECKING:
    # WHY: TYPE_CHECKING guard avoids a runtime circular import.
    # AnalyticsPipeline is only needed as a type annotation — the actual object
    # is passed in at construction time, so no runtime import is required.
    from src.pipeline import AnalyticsPipeline

logger = structlog.get_logger()
_tracer = get_tracer(__name__)


@dataclass
class ConversationTurn:
    """Immutable record of one completed pipeline turn.

    Stored in Conversation._turns; built by ConversationSession._store_turn()
    from the PipelineOutput returned by AnalyticsPipeline.run().
    """

    question: str
    sql: str | None  # WHY: None when status is unanswerable/error — stored so format_context
    #                        can render "(none)" rather than hiding the failed attempt
    answer: str
    status: str  # mirrors PipelineOutput.status: "success" | "unanswerable" | "invalid_sql" | "error"
    timestamp: float  # time.time() — monotonic ordering within a session


class Conversation:
    """Owns per-session turn storage, token accounting, and context formatting.

    WHY: dedicated class (not a dataclass) because the logic for token-aware
    compression and context formatting is non-trivial. No LLM or pipeline
    dependency — those live in ConversationSession, keeping this class purely
    testable with no network calls.

    Compression is NOT triggered inside add_turn() — that is intentional.
    ConversationSession decides when to call needs_summarization() and
    apply_summary() so that summarization logic is centralised and testable
    independently of turn storage.
    """

    def __init__(self) -> None:
        self._turns: list[ConversationTurn] = []
        # WHY: _summary accumulates LLM-generated text from each compression
        # pass.  It grows across multiple passes (newer summaries prepended)
        # so the oldest context is always present, just increasingly compressed.
        self._summary: str = ""

    # ── Turn storage ──────────────────────────────────────────────────────────

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a completed turn.  Never triggers compression.

        WHY: decoupling storage from compression lets ConversationSession
        decide when to summarize, keeping add_turn() simple and testable.
        """
        self._turns.append(turn)

    # ── Token accounting ──────────────────────────────────────────────────────

    def count_context_tokens(self, encoding: str = "cl100k_base") -> int:
        """Return the tiktoken count of the formatted context string.

        WHY: token-based threshold is more accurate than a simple turn-count
        because turns vary in length — a single turn with a long SQL query can
        exceed a budget that a coarser N-turn window would not catch.
        Returns 0 when the conversation is empty.
        """
        text = self.format_context()
        if not text:
            return 0
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))

    def needs_summarization(self, token_limit: int) -> bool:
        """Return True when the formatted context exceeds token_limit tokens."""
        return self.count_context_tokens() > token_limit

    # ── Summarization API ─────────────────────────────────────────────────────

    def get_turns_for_summarization(self, keep_recent: int) -> list[ConversationTurn]:
        """Return turns older than the keep_recent window — candidates for compression.

        WHY: always keep the most recent N turns verbatim so the model has full
        detail on immediately prior context; compress only older turns.
        Returns an empty list when all turns fit within the window.
        """
        if len(self._turns) <= keep_recent:
            return []
        return list(self._turns[: len(self._turns) - keep_recent])

    def apply_summary(
        self, summary_text: str, turns_summarized: list[ConversationTurn]
    ) -> None:
        """Remove the given turns and prepend summary_text to _summary.

        WHY: prepend so the oldest context always appears first in the
        PRIOR TURNS SUMMARY block — chronological order is preserved for the
        LLM reader.  Turns are matched by identity (id()) so equal-looking
        turns from different sessions cannot accidentally be removed.
        """
        ids_to_remove = {id(t) for t in turns_summarized}
        self._turns = [t for t in self._turns if id(t) not in ids_to_remove]
        # WHY: strip + join avoids leading/trailing blank lines when either
        # summary_text or _summary is empty
        parts = [p for p in (summary_text.strip(), self._summary.strip()) if p]
        self._summary = "\n\n".join(parts)

    # ── Context formatting ────────────────────────────────────────────────────

    def format_context(self) -> str:
        """Render conversation history as a prompt-ready string.

        Format:
            PRIOR TURNS SUMMARY:
            <LLM-generated text>

            RECENT TURNS:
            Turn 1:
              Question: "..."
              SQL: SELECT ...
              Answer: "..."

        Returns "" when the conversation is empty.
        """
        if self.is_empty():
            return ""
        parts: list[str] = []
        if self._summary:
            parts.append(f"PRIOR TURNS SUMMARY:\n{self._summary}")
        if self._turns:
            lines: list[str] = []
            for i, turn in enumerate(self._turns, 1):
                sql_display = turn.sql if turn.sql is not None else "(none)"
                lines.append(
                    f"Turn {i}:\n"
                    f"  Question: {turn.question}\n"
                    f"  SQL: {sql_display}\n"
                    f"  Answer: {turn.answer}"
                )
            parts.append("RECENT TURNS:\n" + "\n\n".join(lines))
        return "\n\n".join(parts)

    # ── Predicates / properties ───────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True when there are no turns and no summary text."""
        return not self._turns and not self._summary

    @property
    def recent_turns(self) -> list[ConversationTurn]:
        """Snapshot of the current in-window turns (does not include summarized turns)."""
        return list(self._turns)

    @property
    def summary(self) -> str:
        """Accumulated LLM-generated summary text of compressed older turns."""
        return self._summary


class ConversationSession:
    """Stateful wrapper around AnalyticsPipeline for a single conversation session.

    WHY: centralising conversation management here keeps AnalyticsPipeline
    stateless (no Conversation dependency) and the server thin (no turn-
    management logic).  One ConversationSession per active client session;
    all sessions share a single AnalyticsPipeline instance (schema context is
    loaded once at startup).

    Lifecycle:
      1. Server creates a ConversationSession when a new session_id is seen.
      2. For each request the server calls session.run(question).
      3. Internally: summarize if needed → classify intent → call pipeline → store turn.
      4. When conversation_history_enabled=False the run() is a pure passthrough.
    """

    def __init__(
        self,
        pipeline: "AnalyticsPipeline",
        config: PipelineConfig,
        session_id: str | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._conversation = Conversation()
        self._session_id = session_id
        # WHY: reuse the pipeline's llm client — avoids a second OpenRouter
        # connection and keeps all LLM calls (including conv ops) through the
        # same retry / token-stats machinery
        self._llm = pipeline.llm

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, question: str, request_id: str | None = None) -> PipelineOutput:
        """Run the pipeline with conversation context management.

        Dispatch logic:
          - feature disabled OR empty history → passthrough to pipeline (no LLM overhead)
          - data_question → answer_from_context (SQL cycle bypassed entirely)
          - follow_up     → pipeline.run() with formatted context injected
          - new_query     → pipeline.run() with no context (fresh SQL generation)

        A conversation.session.run OTel span wraps the full turn; pipeline.run
        becomes its child automatically via context propagation — no changes to
        pipeline.py are required.
        """
        with _tracer.start_as_current_span("conversation.session.run") as span:
            span.set_attribute("conv.session_id", self._session_id or "")
            span.set_attribute("conv.history_enabled", self._config.conversation_history_enabled)

            if not self._config.conversation_history_enabled or self._conversation.is_empty():
                # WHY: zero overhead for the disabled case and the first turn in a session
                span.set_attribute("conv.intent", "passthrough")
                span.set_attribute("conv.path", "passthrough")
                result = self._pipeline.run(question, request_id, conversation_context="")
                result = dataclasses.replace(result, intent="passthrough")
                self._store_turn(question, result)
                return result

            self._maybe_summarize()

            if self._config.intent_prediction_enabled:
                intent_output: IntentClassificationOutput = self._llm.classify_intent(
                    question, self._conversation
                )
                logger.debug(
                    "Intent classification completed",
                    intent=intent_output.intent,
                    error=intent_output.error,
                )
                intent = intent_output.intent
            else:
                # WHY: when prediction is disabled always treat as follow_up so context is injected
                intent = "follow_up"

            span.set_attribute("conv.intent", intent)

            if intent == "data_question":
                span.set_attribute("conv.path", "data_question")
                result = self._answer_from_context(question, request_id, intent=intent)
            else:
                ctx = self._conversation.format_context() if intent == "follow_up" else ""
                span.set_attribute("conv.path", "pipeline")
                result = self._pipeline.run(question, request_id, conversation_context=ctx)
                result = dataclasses.replace(result, intent=intent)

            self._store_turn(question, result)
            return result

    def reset(self) -> None:
        """Clear conversation state — start a fresh conversation in this session."""
        self._conversation = Conversation()

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    # ── Private helpers ───────────────────────────────────────────────────────

    def _answer_from_context(self, question: str, request_id: str | None, intent: str = "data_question") -> PipelineOutput:
        """Answer directly from conversation context — bypasses the SQL cycle entirely.

        WHY: data_question intent means the answer exists in prior results already
        captured in the conversation context.  Running a new SQL cycle is wasteful
        and may not retrieve the same subset the user is asking about.
        On LLM error, answer_from_context returns an AnswerGenerationOutput with
        error set; we propagate that as status="error" in the PipelineOutput.
        """
        ctx = self._conversation.format_context()
        answer_output: AnswerGenerationOutput = self._llm.answer_from_context(question, ctx)
        status = "error" if answer_output.error else "success"
        # WHY: sql=None because no SQL was generated; all SQL-stage outputs are
        # empty stubs — the pipeline contract requires these fields to be present.
        return PipelineOutput(
            status=status,
            question=question,
            request_id=request_id,
            sql_generation=SQLGenerationOutput(
                sql=None, answerable=None, timing_ms=0.0, llm_stats={}
            ),
            sql_validation=SQLValidationOutput(is_valid=True, validated_sql=None),
            sql_execution=SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0),
            answer_generation=answer_output,
            sql=None,
            rows=[],
            answer=answer_output.answer,
            result_validation=ResultValidationOutput(),
            intent=intent,
        )

    def _maybe_summarize(self) -> None:
        """Trigger LLM summarization if the formatted context exceeds the token limit.

        WHY: called before intent classification so the context passed to the
        LLM for classification is already within budget.  Non-blocking: a
        summarization failure retains the original turns and logs a warning —
        the pipeline call still proceeds.
        """
        if not self._conversation.needs_summarization(
            self._config.conversation_context_token_limit
        ):
            return

        turns_to_compress = self._conversation.get_turns_for_summarization(
            keep_recent=self._config.conversation_history_window
        )
        if not turns_to_compress and self._conversation.recent_turns:
            # WHY: all turns fall within the window but tokens still exceed the
            # limit (e.g. window=5 with only 1 very long turn). Fall back to
            # keeping only the single most recent turn and compressing the rest
            # so the context stays bounded regardless of window size.
            n = len(self._conversation.recent_turns)
            turns_to_compress = self._conversation.get_turns_for_summarization(
                keep_recent=max(0, n - 1)
            )
        if not turns_to_compress:
            return

        summarization_output: SummarizationOutput = self._llm.summarize_turns(
            turns_to_compress
        )
        if summarization_output.error:
            logger.warning(
                "Conversation summarization failed, retaining original turns",
                error=summarization_output.error,
                turns_count=len(turns_to_compress),
            )
            return

        self._conversation.apply_summary(
            summarization_output.summary, turns_to_compress
        )
        logger.debug(
            "Conversation summarized",
            turns_compressed=len(turns_to_compress),
            summary_length=len(summarization_output.summary),
        )

    def _store_turn(self, question: str, result: PipelineOutput) -> None:
        """Append a ConversationTurn built from the pipeline result.

        WHY: always stores the turn, even on error/unanswerable, so the
        conversation history is complete and the model can reason about why
        a prior question failed.
        """
        turn = ConversationTurn(
            question=question,
            sql=result.sql,
            answer=result.answer,
            status=result.status,
            timestamp=time.time(),
        )
        self._conversation.add_turn(turn)
