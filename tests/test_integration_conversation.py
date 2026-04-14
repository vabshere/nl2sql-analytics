"""Integration tests for multi-turn conversation using the real OpenRouter API.

These tests exercise ConversationSession end-to-end with no mocking:
real LLM calls, real SQLite DB, real intent classification and summarization.

Run with:
    OPENROUTER_API_KEY=sk-or-... pytest -m integration -v

Skipped automatically when OPENROUTER_API_KEY, gaming DB, or schema metadata DB is absent.

WHY test via ConversationSession directly (not HTTP server):
  Same pattern as test_public.py for AnalyticsPipeline — tests the full
  conversation + LLM flow at the right level without server startup overhead.

WHY assert structure not content:
  LLM output is non-deterministic; we check status, intent, turn count,
  and that expected known values appear in answers — not exact strings.

Known DB facts (gaming_mental_health, 1 000 000 rows):
  - Total rows            : 1 000 000
  - Distinct genders      : 3  (Female 479 579, Male 480 581, Other 39 840)
  - Female %              : ~47.96%  (→ assert "47" or "48" in answer)
  - avg daily_gaming_hours: 4.0
  - max daily_gaming_hours: 30.42
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.gaming_csv_to_db import DEFAULT_DB_PATH
    from src.config import PipelineConfig
    from src.conversation import ConversationSession
    from src.pipeline import AnalyticsPipeline
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


_METADATA_DB = PROJECT_ROOT / "data" / "schema_metadata.sqlite"

_SKIP = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY")
    or not DEFAULT_DB_PATH.exists()
    or not _METADATA_DB.exists(),
    reason=(
        "Integration tests require OPENROUTER_API_KEY, "
        f"{DEFAULT_DB_PATH} (gaming DB), and {_METADATA_DB} (schema metadata DB)."
    ),
)


def _make_session() -> ConversationSession:
    """Build a real ConversationSession wired to the real DB and LLM."""
    config = PipelineConfig(
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        db_path=DEFAULT_DB_PATH,
        metadata_db_path=_METADATA_DB,
        conversation_history_enabled=True,
        intent_prediction_enabled=True,
    )
    pipeline = AnalyticsPipeline(config=config)
    return ConversationSession(pipeline=pipeline, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_SKIP
@pytest.mark.integration
def test_single_turn_with_history_enabled():
    """Baseline: one turn returns the correct gender count and is stored.

    DB truth: 3 distinct genders (Female, Male, Other).
    First turn on empty history → intent must be "passthrough" (no classification).
    """
    session = _make_session()
    output = session.run("How many distinct genders are in the dataset?")

    assert output.status == "success", f"Unexpected status: {output.answer}"
    assert output.intent == "passthrough"  # empty history → no classification
    assert "3" in output.answer, f"Expected '3' in answer, got: {output.answer!r}"
    assert len(session._conversation.recent_turns) == 1


@_SKIP
@pytest.mark.integration
def test_two_turn_follow_up_builds_history():
    """Two turns in the same session accumulate history; second turn is classified.

    Turn 1: ask for distinct genders → answer must list all three gender values.
    Turn 2: ask for their count → should classify as follow_up or data_question;
            answer must contain "3".
    """
    session = _make_session()

    out1 = session.run("What are the distinct genders in the dataset?")
    assert out1.status == "success", f"Turn 1 failed: {out1.answer}"
    assert out1.intent == "passthrough"  # empty history on first call
    # At least one gender name should appear; LLM may list them differently
    assert any(g in out1.answer for g in ("Female", "Male", "Other")), (
        f"Expected at least one gender name in turn-1 answer: {out1.answer!r}"
    )

    out2 = session.run("How many of those genders are there in total?")
    assert out2.status == "success", f"Turn 2 failed: {out2.answer}"
    # Second turn must be classified (history is non-empty)
    assert out2.intent in ("follow_up", "data_question"), (
        f"Expected follow_up or data_question, got {out2.intent!r}"
    )
    assert "3" in out2.answer, f"Expected '3' in turn-2 answer, got: {out2.answer!r}"
    assert len(session._conversation.recent_turns) == 2


@_SKIP
@pytest.mark.integration
def test_new_session_has_no_history():
    """A fresh session starts empty; first turn is passthrough and returns row count.

    DB truth: 1 000 000 total rows.
    """
    session = _make_session()
    assert session._conversation.is_empty()

    output = session.run("How many rows are in the dataset?")

    assert output.status == "success", f"Unexpected status: {output.answer}"
    assert output.intent == "passthrough"
    # Accept both "1000000" and "1,000,000" formatting
    assert "1000000" in output.answer.replace(",", ""), (
        f"Expected row count in answer, got: {output.answer!r}"
    )
    assert not session._conversation.is_empty()


@_SKIP
@pytest.mark.integration
def test_data_question_bypasses_pipeline():
    """After a turn with rows, a follow-up question is handled without error.

    WHY lax: intent classification is non-deterministic — the LLM may route as
    data_question (answer from context, sql=None) or follow_up (new SQL query).
    Both are valid. We assert only that the session succeeds and produces an answer,
    and that the question is not misclassified as an unrelated new_query.
    """
    session = _make_session()

    out1 = session.run("What are the distinct genders in the dataset?")
    assert out1.status == "success"

    out2 = session.run("From the genders you just listed, which one appears most?")

    assert out2.status == "success", f"Expected success, got {out2.status!r}: {out2.answer}"
    assert out2.intent in ("data_question", "follow_up"), (
        f"Expected data_question or follow_up, got {out2.intent!r}"
    )
    assert out2.answer, "Expected a non-empty answer"


# ---------------------------------------------------------------------------
# 10-turn end-to-end conversation
# ---------------------------------------------------------------------------


def _assert_data_question_or_follow_up(out, turn: int) -> None:
    """Soft assertion for turns expected to be data_question.

    WHY: intent classification is non-deterministic — the LLM may return
    follow_up instead of data_question for questions that reference prior results.
    Both are valid routing decisions; we accept either and assert accordingly.
    """
    assert out.status == "success", f"T{turn} failed: {out.answer}"
    if out.intent == "data_question":
        assert out.sql is None, f"T{turn} data_question must have sql=None"
    else:
        # Accepted fallback: follow_up runs the SQL cycle but still answers correctly
        assert out.intent == "follow_up", (
            f"T{turn} expected data_question or follow_up, got {out.intent!r}"
        )
        assert out.sql is not None, f"T{turn} follow_up must have sql"


@_SKIP
@pytest.mark.integration
def test_ten_turn_conversation_covers_all_intents():
    """10-turn conversation exercising every intent type and routing path.

    Turn layout
    -----------
    Phase 1 — gender exploration
      T1  passthrough   : first turn on empty history → gender count
      T2  follow_up     : list each gender with its count
      T3  data_question : which gender has the fewest rows? (soft: accept follow_up)

    Phase 2 — gaming hours
      T4  new_query     : average daily_gaming_hours
      T5  follow_up     : maximum daily_gaming_hours
      T6  follow_up     : how much higher is the max than the average?

    Phase 3 — dataset scale
      T7  new_query     : total row count
      T8  follow_up     : percentage of Female users

    Phase 4 — new topic with data_question close
      T9  new_query     : average stress_level (new unrelated metric)
      T10 data_question : is that stress level considered high? (soft: accept follow_up)

    DB ground truth used in assertions
    -----------------------------------
    - Distinct genders      : 3
    - Gender names          : Female, Male, Other
    - avg daily_gaming_hours: 4.0  → assert "4" in answer
    - max daily_gaming_hours: 30.42 → assert "30" in answer
    - Total rows            : 1 000 000
    - Female %              : ~47.96% → assert "47" or "48" in answer
    """
    session = _make_session()

    # ── Phase 1: gender exploration ───────────────────────────────────────────

    # T1 — passthrough (empty history, no classification)
    out1 = session.run("How many distinct genders are in the dataset?")
    assert out1.status == "success", f"T1 failed: {out1.answer}"
    assert out1.intent == "passthrough"
    assert "3" in out1.answer, f"T1: expected '3' in {out1.answer!r}"

    # T2 — follow_up: refine T1 by listing genders with counts
    out2 = session.run("List each of those genders and how many rows each has.")
    assert out2.status == "success", f"T2 failed: {out2.answer}"
    assert out2.intent in ("follow_up", "new_query"), f"T2 expected follow_up/new_query, got {out2.intent!r}"
    # At least one of the known gender names must appear; LLM may not list all three
    assert any(g in out2.answer for g in ("Female", "Male", "Other")), (
        f"T2: expected at least one gender name in {out2.answer!r}"
    )

    # T3 — data_question: answer is already in T2's rows (soft assertion)
    out3 = session.run("From the counts you just listed, which gender has the fewest rows?")
    _assert_data_question_or_follow_up(out3, turn=3)

    # ── Phase 2: gaming hours ─────────────────────────────────────────────────

    # T4 — new_query: entirely new metric, no reference to prior turns
    out4 = session.run("What is the average daily_gaming_hours across all users?")
    assert out4.status == "success", f"T4 failed: {out4.answer}"
    assert out4.intent in ("new_query", "follow_up"), f"T4 intent: {out4.intent!r}"
    # Real avg is 3.9992555; SQL may round to 4.0 or report precise value.
    # Accept any representation containing "3.9" or "4" (digit boundary check).
    assert any(s in out4.answer for s in ("4.0", "3.9", "3.99", " 4 ", "≈4", "about 4")), (
        f"T4: expected avg ~4.0 in {out4.answer!r}"
    )

    # T5 — follow_up: extend T4 with max value
    out5 = session.run("What is the maximum daily_gaming_hours?")
    assert out5.status == "success", f"T5 failed: {out5.answer}"
    assert out5.intent in ("follow_up", "new_query"), f"T5 expected follow_up/new_query, got {out5.intent!r}"
    # max is 30.42 — "30" will always appear
    assert "30" in out5.answer, f"T5: expected max ~30.42 in {out5.answer!r}"

    # T6 — follow_up: extend further, comparing max to average
    out6 = session.run("How much higher is that maximum than the average?")
    assert out6.status == "success", f"T6 failed: {out6.answer}"
    assert out6.intent in ("follow_up", "new_query"), f"T6 expected follow_up/new_query, got {out6.intent!r}"

    # ── Phase 3: dataset scale ────────────────────────────────────────────────

    # T7 — new_query: completely new question about dataset size
    out7 = session.run("How many rows are in the dataset in total?")
    assert out7.status == "success", f"T7 failed: {out7.answer}"
    assert out7.intent in ("new_query", "follow_up"), f"T7 intent: {out7.intent!r}"
    assert "1000000" in out7.answer.replace(",", ""), (
        f"T7: expected 1000000 in {out7.answer!r}"
    )

    # T8 — follow_up: extend T7 with gender breakdown percentage
    out8 = session.run("What percentage of those rows belong to Female users?")
    assert out8.status == "success", f"T8 failed: {out8.answer}"
    assert out8.intent in ("follow_up", "new_query"), f"T8 expected follow_up/new_query, got {out8.intent!r}"
    # Female = 479579 / 1000000 = 47.96%; accept "47" or "48" depending on rounding
    assert "47" in out8.answer or "48" in out8.answer, (
        f"T8: expected ~47-48% in {out8.answer!r}"
    )

    # ── Phase 4: new topic with data_question close ───────────────────────────

    # T9 — new_query: pivot to an entirely different metric
    out9 = session.run("What is the average stress_level for all users?")
    assert out9.status == "success", f"T9 failed: {out9.answer}"
    assert out9.intent in ("new_query", "follow_up"), f"T9 intent: {out9.intent!r}"

    # T10 — data_question: ask about the value just returned by T9 (soft assertion)
    out10 = session.run(
        "Based on the average stress level you just calculated, "
        "is that value closer to 0 or to 10?"
    )
    _assert_data_question_or_follow_up(out10, turn=10)

    # ── Final state check ─────────────────────────────────────────────────────
    assert len(session._conversation.recent_turns) == 10
