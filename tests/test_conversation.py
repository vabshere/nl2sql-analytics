"""Tests for Conversation, ConversationSession, and ConversationTurn.

TDD: these tests are written before the implementation.  They cover:
  - Conversation: turn storage, token counting, summarization API, context formatting
  - ConversationSession: passthrough when disabled, context injection, summarization trigger,
    intent classification, turn storage after each run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.conversation import Conversation, ConversationSession, ConversationTurn
    from src.config import PipelineConfig
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
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ZERO_STATS: dict[str, Any] = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "model": "stub",
}


def _turn(
    question: str = "How many users?",
    sql: str | None = "SELECT COUNT(*) FROM t",
    answer: str = "42",
    status: str = "success",
) -> ConversationTurn:
    return ConversationTurn(
        question=question, sql=sql, answer=answer, status=status, timestamp=time.time()
    )


def _pipeline_output(
    status: str = "success",
    sql: str | None = "SELECT COUNT(*) FROM t",
    answer: str = "42",
) -> PipelineOutput:
    return PipelineOutput(
        status=status,
        question="How many users?",
        request_id=None,
        sql_generation=SQLGenerationOutput(
            sql=sql, answerable=True, timing_ms=0.0, llm_stats=dict(_ZERO_STATS)
        ),
        sql_validation=SQLValidationOutput(is_valid=True, validated_sql=sql),
        sql_execution=SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0),
        answer_generation=AnswerGenerationOutput(
            answer=answer, timing_ms=0.0, llm_stats=dict(_ZERO_STATS)
        ),
        sql=sql,
        rows=[],
        answer=answer,
        result_validation=ResultValidationOutput(),
    )


def _make_session(
    *,
    enabled: bool = True,
    window: int = 5,
    token_limit: int = 2000,
    intent_enabled: bool = True,
    pipeline_result: PipelineOutput | None = None,
) -> tuple[ConversationSession, MagicMock, MagicMock]:
    """Return (session, mock_pipeline, mock_llm).

    mock_llm is pre-wired with sensible defaults:
      summarize_turns  → SummarizationOutput(summary="", no error)
      classify_intent  → IntentClassificationOutput(intent="follow_up", ...)
    """
    mock_llm = MagicMock()
    mock_llm.summarize_turns.return_value = SummarizationOutput(
        summary="summary text", llm_stats=dict(_ZERO_STATS)
    )
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="follow_up",
        reason="references prior result",
        llm_stats=dict(_ZERO_STATS),
    )

    mock_pipeline = MagicMock()
    mock_pipeline.llm = mock_llm
    mock_pipeline.run.return_value = pipeline_result or _pipeline_output()

    config = PipelineConfig(
        openrouter_api_key="test-key",
        conversation_history_enabled=enabled,
        conversation_history_window=window,
        conversation_context_token_limit=token_limit,
        intent_prediction_enabled=intent_enabled,
    )
    session = ConversationSession(pipeline=mock_pipeline, config=config)
    return session, mock_pipeline, mock_llm


# ===========================================================================
# Conversation — turn storage
# ===========================================================================


def test_add_turn_appends():
    conv = Conversation()
    t1 = _turn(question="Q1")
    t2 = _turn(question="Q2")
    conv.add_turn(t1)
    conv.add_turn(t2)
    assert conv.recent_turns == [t1, t2]


def test_add_turn_does_not_auto_compress():
    """add_turn must never modify _summary — compression is ConversationSession's job."""
    conv = Conversation()
    for i in range(10):
        conv.add_turn(_turn(question=f"Q{i}"))
    assert conv.summary == ""
    assert len(conv.recent_turns) == 10


# ===========================================================================
# Conversation — token counting
# ===========================================================================


def test_count_context_tokens_empty():
    """Empty conversation has 0 tokens."""
    conv = Conversation()
    assert conv.count_context_tokens() == 0


def test_count_context_tokens_positive():
    """Non-empty conversation has a positive token count."""
    conv = Conversation()
    conv.add_turn(_turn())
    assert conv.count_context_tokens() > 0


def test_needs_summarization_below_limit():
    conv = Conversation()
    conv.add_turn(_turn())
    tokens = conv.count_context_tokens()
    assert not conv.needs_summarization(token_limit=tokens + 100)


def test_needs_summarization_above_limit():
    conv = Conversation()
    conv.add_turn(_turn())
    tokens = conv.count_context_tokens()
    assert conv.needs_summarization(token_limit=tokens - 1)


# ===========================================================================
# Conversation — get_turns_for_summarization
# ===========================================================================


def test_get_turns_for_summarization_within_window():
    """When len(turns) <= keep_recent, nothing should be returned."""
    conv = Conversation()
    for i in range(3):
        conv.add_turn(_turn(question=f"Q{i}"))
    assert conv.get_turns_for_summarization(keep_recent=5) == []


def test_get_turns_for_summarization_overflow():
    """Returns oldest turns beyond keep_recent window."""
    conv = Conversation()
    turns = [_turn(question=f"Q{i}") for i in range(7)]
    for t in turns:
        conv.add_turn(t)
    result = conv.get_turns_for_summarization(keep_recent=3)
    # oldest 4 turns (7 - 3 = 4) returned
    assert result == turns[:4]
    # most recent 3 NOT included
    assert turns[4] not in result
    assert turns[5] not in result
    assert turns[6] not in result


def test_get_turns_for_summarization_exact_window():
    """When len(turns) == keep_recent exactly, nothing returned."""
    conv = Conversation()
    for i in range(5):
        conv.add_turn(_turn(question=f"Q{i}"))
    assert conv.get_turns_for_summarization(keep_recent=5) == []


# ===========================================================================
# Conversation — apply_summary
# ===========================================================================


def test_apply_summary_removes_turns():
    conv = Conversation()
    turns = [_turn(question=f"Q{i}") for i in range(5)]
    for t in turns:
        conv.add_turn(t)
    to_summarize = turns[:3]
    conv.apply_summary("compressed", to_summarize)
    remaining = conv.recent_turns
    assert len(remaining) == 2
    assert turns[3] in remaining
    assert turns[4] in remaining
    assert turns[0] not in remaining


def test_apply_summary_sets_summary_text():
    conv = Conversation()
    conv.add_turn(_turn())
    conv.apply_summary("my summary", conv.recent_turns)
    assert "my summary" in conv.summary


def test_apply_summary_prepends_to_existing_summary():
    """Second compression prepends new text before existing summary."""
    conv = Conversation()
    for i in range(4):
        conv.add_turn(_turn(question=f"Q{i}"))
    first_batch = conv.recent_turns[:2]
    conv.apply_summary("first summary", first_batch)
    # Add more turns and summarize again
    for i in range(4, 6):
        conv.add_turn(_turn(question=f"Q{i}"))
    second_batch = conv.recent_turns[:2]
    conv.apply_summary("second summary", second_batch)
    # second summary appears before first
    assert conv.summary.index("second summary") < conv.summary.index("first summary")


# ===========================================================================
# Conversation — format_context
# ===========================================================================


def test_format_context_empty():
    assert Conversation().format_context() == ""


def test_format_context_recent_only():
    conv = Conversation()
    conv.add_turn(_turn(question="What is X?", answer="X is 5"))
    ctx = conv.format_context()
    assert "RECENT TURNS" in ctx
    assert "What is X?" in ctx
    assert "X is 5" in ctx
    # No summary section when _summary is empty
    assert "PRIOR TURNS SUMMARY" not in ctx


def test_format_context_with_summary():
    conv = Conversation()
    conv.add_turn(_turn(question="Q old", answer="old ans"))
    to_summarize = conv.recent_turns[:]
    conv.apply_summary("Old stuff happened", to_summarize)
    conv.add_turn(_turn(question="Q recent", answer="recent ans"))

    ctx = conv.format_context()
    assert "PRIOR TURNS SUMMARY" in ctx
    assert "Old stuff happened" in ctx
    assert "RECENT TURNS" in ctx
    assert "Q recent" in ctx


def test_format_context_null_sql_turn():
    """sql=None renders as '(none)', not 'None'."""
    conv = Conversation()
    conv.add_turn(_turn(sql=None))
    assert "(none)" in conv.format_context()
    assert "None" not in conv.format_context()


def test_format_context_summary_before_recent():
    """PRIOR TURNS SUMMARY section always precedes RECENT TURNS section."""
    conv = Conversation()
    conv.add_turn(_turn(question="old"))
    conv.apply_summary("summary text", conv.recent_turns[:])
    conv.add_turn(_turn(question="new"))
    ctx = conv.format_context()
    assert ctx.index("PRIOR TURNS SUMMARY") < ctx.index("RECENT TURNS")


# ===========================================================================
# Conversation — is_empty
# ===========================================================================


def test_is_empty_true_initially():
    assert Conversation().is_empty()


def test_is_empty_false_after_add():
    conv = Conversation()
    conv.add_turn(_turn())
    assert not conv.is_empty()


def test_is_empty_false_with_summary_only():
    """Conversation with only a summary (no recent turns) is not empty."""
    conv = Conversation()
    conv.add_turn(_turn())
    conv.apply_summary("summary", conv.recent_turns[:])
    # all turns summarized — recent_turns now empty, summary non-empty
    assert not conv.is_empty()


# ===========================================================================
# ConversationSession — passthrough / disabled
# ===========================================================================


def test_session_run_passthrough_when_disabled():
    """When conversation_history_enabled=False, pipeline.run() gets conversation_context=""."""
    session, mock_pipeline, mock_llm = _make_session(enabled=False)
    session.run("Q?")
    mock_pipeline.run.assert_called_once()
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") == ""
    mock_llm.classify_intent.assert_not_called()
    mock_llm.summarize_turns.assert_not_called()


def test_session_run_passthrough_empty_history():
    """Empty conversation (first turn) passes no context to the pipeline."""
    session, mock_pipeline, mock_llm = _make_session(enabled=True)
    session.run("First question")
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") == ""
    mock_llm.classify_intent.assert_not_called()


# ===========================================================================
# ConversationSession — turn storage
# ===========================================================================


def test_session_run_stores_turn_after_success():
    session, _, _ = _make_session()
    assert session.conversation.is_empty()
    session.run("Q?")
    assert len(session.conversation.recent_turns) == 1
    turn = session.conversation.recent_turns[0]
    assert turn.question == "Q?"
    assert turn.status == "success"
    assert turn.sql == "SELECT COUNT(*) FROM t"
    assert turn.answer == "42"


def test_session_run_stores_turn_after_error():
    """Even unanswerable/error runs store a turn so history is complete."""
    result = _pipeline_output(status="unanswerable", sql=None, answer="Cannot answer.")
    session, _, _ = _make_session(pipeline_result=result)
    session.run("Delete everything")
    turn = session.conversation.recent_turns[0]
    assert turn.status == "unanswerable"
    assert turn.sql is None


def test_session_run_second_turn_appended():
    session, _, _ = _make_session()
    session.run("Q1")
    session.run("Q2")
    assert len(session.conversation.recent_turns) == 2


# ===========================================================================
# ConversationSession — intent classification
# ===========================================================================


def test_session_run_follow_up_injects_context():
    """follow_up intent → conversation_context passed to pipeline.run()."""
    session, mock_pipeline, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="follow_up", reason="refs prior", llm_stats=dict(_ZERO_STATS)
    )
    session.run("Q1")  # first turn (no context yet)
    session.run("Q2")  # second turn — history exists, intent=follow_up
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") != ""


def test_session_run_new_query_no_context():
    """new_query intent → conversation_context is ""."""
    session, mock_pipeline, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="new_query", reason="new topic", llm_stats=dict(_ZERO_STATS)
    )
    session.run("Q1")  # populates history
    session.run("Q2 unrelated")
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") == ""


def test_session_run_intent_error_injects_context():
    """When classify_intent returns error (intent=follow_up fallback), context IS injected."""
    session, mock_pipeline, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="follow_up",  # error fallback
        reason="",
        llm_stats=dict(_ZERO_STATS),
        error="LLM call failed",
    )
    session.run("Q1")
    session.run("Q2")
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") != ""


def test_session_run_intent_disabled_always_injects_context():
    """When intent_prediction_enabled=False, context always injected (no classification call)."""
    session, mock_pipeline, mock_llm = _make_session(intent_enabled=False)
    session.run("Q1")
    session.run("Q2")
    mock_llm.classify_intent.assert_not_called()
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("conversation_context", "") != ""


# ===========================================================================
# ConversationSession — data_question intent
# ===========================================================================


def _data_question_answer() -> AnswerGenerationOutput:
    return AnswerGenerationOutput(answer="The max was 10.", timing_ms=0.0, llm_stats=dict(_ZERO_STATS))


def test_session_run_data_question_bypasses_pipeline():
    """data_question intent → pipeline.run() must NOT be called."""
    session, mock_pipeline, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="data_question", reason="asking about prior rows", llm_stats=dict(_ZERO_STATS)
    )
    mock_llm.answer_from_context.return_value = _data_question_answer()
    session.run("Q1")  # populate history
    mock_pipeline.run.reset_mock()
    session.run("What was the max?")
    mock_pipeline.run.assert_not_called()


def test_session_run_data_question_calls_answer_from_context():
    """data_question intent → answer_from_context IS called with question and context string."""
    session, _, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="data_question", reason="asking about prior rows", llm_stats=dict(_ZERO_STATS)
    )
    mock_llm.answer_from_context.return_value = _data_question_answer()
    session.run("Q1")  # populate history
    session.run("What was the max?")
    mock_llm.answer_from_context.assert_called_once()
    args, kwargs = mock_llm.answer_from_context.call_args
    question_arg = args[0] if args else kwargs.get("question", "")
    assert question_arg == "What was the max?"


def test_session_run_data_question_stores_turn():
    """data_question result is stored in conversation history with sql=None."""
    session, _, mock_llm = _make_session()
    mock_llm.classify_intent.return_value = IntentClassificationOutput(
        intent="data_question", reason="asking about prior rows", llm_stats=dict(_ZERO_STATS)
    )
    mock_llm.answer_from_context.return_value = _data_question_answer()
    session.run("Q1")
    session.run("What was the max?")
    assert len(session.conversation.recent_turns) == 2
    last_turn = session.conversation.recent_turns[1]
    assert last_turn.answer == "The max was 10."
    assert last_turn.sql is None


# ===========================================================================
# ConversationSession — summarization
# ===========================================================================


def test_session_run_summarization_not_triggered_below_limit():
    """When token count is below limit, summarize_turns must NOT be called."""
    session, _, mock_llm = _make_session(token_limit=99999)
    session.run("Q1")
    session.run("Q2")
    mock_llm.summarize_turns.assert_not_called()


def test_session_run_summarization_triggered_on_overflow():
    """When token count exceeds the limit, summarize_turns IS called."""
    session, _, mock_llm = _make_session(token_limit=1)  # almost certainly exceeded
    session.run("Q1")  # first turn: stored but no history yet during run
    session.run("Q2")  # second run: history exists + tokens > 1 → triggers summarization
    mock_llm.summarize_turns.assert_called_once()


def test_session_run_summarization_error_continues():
    """Summarization error must not abort the run — pipeline.run() still called."""
    session, mock_pipeline, mock_llm = _make_session(token_limit=1)
    mock_llm.summarize_turns.return_value = SummarizationOutput(
        summary="", llm_stats=dict(_ZERO_STATS), error="LLM down"
    )
    session.run("Q1")
    session.run("Q2")  # summarization errors — should not raise
    mock_pipeline.run.assert_called()
    # apply_summary should NOT have been called — turns retained as-is
    assert len(session.conversation.recent_turns) == 2


def test_session_run_summarization_compresses_overflow_turns():
    """After successful summarization, old turns are replaced by summary text."""
    session, _, mock_llm = _make_session(token_limit=1, window=1)
    mock_llm.summarize_turns.return_value = SummarizationOutput(
        summary="Q1 asked about users, answer was 42",
        llm_stats=dict(_ZERO_STATS),
    )
    session.run("Q1")
    session.run("Q2")  # triggers summarization; Q1 turn compressed
    assert "Q1 asked about users" in session.conversation.summary


# ===========================================================================
# ConversationSession — reset
# ===========================================================================


def test_session_reset_clears_conversation():
    session, _, _ = _make_session()
    session.run("Q1")
    assert not session.conversation.is_empty()
    session.reset()
    assert session.conversation.is_empty()
