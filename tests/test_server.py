"""Server session-management and input-validation tests.

Covers:
- Session creation on first request, reuse on subsequent requests.
- Question length and empty-input validation (HTTP 422).
ConversationSession is mocked — session routing logic only, not conversation internals.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient
    from src.server import app
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


def _stub_pipeline_output():
    out = MagicMock()
    out.status = "success"
    out.answer = "42"
    out.sql = "SELECT age FROM gaming_mental_health"
    return out


@pytest.fixture
def client():
    """TestClient with PipelineConfig, AnalyticsPipeline, and ConversationSession mocked.

    WHY: server lifespan needs real DB paths; mocking the pipeline avoids that
    dependency and keeps these tests focused on session routing only.
    """
    mock_pipeline = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = _stub_pipeline_output()

    mock_config = MagicMock()
    mock_config.max_question_length = 2000

    with (
        patch("src.server.PipelineConfig", return_value=mock_config),
        patch("src.server.AnalyticsPipeline", return_value=mock_pipeline),
        patch("src.server.ConversationSession", return_value=mock_session) as mock_session_cls,
    ):
        with TestClient(app) as c:
            c._mock_session_cls = mock_session_cls
            c._mock_session = mock_session
            yield c


def test_run_no_session_id_creates_new_session(client):
    """A request without session_id creates a new session and returns a session_id."""
    resp = client.post("/run", json={"question": "How many users?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"]  # non-empty
    # ConversationSession was constructed once
    assert client._mock_session_cls.call_count == 1


def test_run_existing_session_id_reuses_session(client):
    """Two requests with the same session_id reuse the same ConversationSession."""
    resp1 = client.post("/run", json={"question": "How many users?"})
    session_id = resp1.json()["session_id"]

    resp2 = client.post("/run", json={"question": "Break that down by gender", "session_id": session_id})
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == session_id

    # Session constructed only once; second request reuses it
    assert client._mock_session_cls.call_count == 1
    assert client._mock_session.run.call_count == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_question_exceeds_max_length_returns_422(client):
    """Question longer than max_question_length must be rejected with HTTP 422."""
    long_question = "x" * 2001
    resp = client.post("/run", json={"question": long_question})
    assert resp.status_code == 422
    assert "2000" in resp.json()["detail"]


def test_question_at_max_length_accepted(client):
    """Question exactly at max_question_length must be accepted."""
    question = "x" * 2000
    resp = client.post("/run", json={"question": question})
    assert resp.status_code == 200


def test_empty_question_returns_422(client):
    """Whitespace-only question must be rejected with HTTP 422."""
    resp = client.post("/run", json={"question": "   "})
    assert resp.status_code == 422
