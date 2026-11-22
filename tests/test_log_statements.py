import logging
import os
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api import app
from chatbot import TOOL_ERROR_MESSAGE, log_tool_error
from config import DEFAULT_MODEL_NAME, get_log_user_text
from tools import calculator_tool, get_current_datetime


# Checks that the user's text is logged only when LOG_USER_TEXT is on.
@pytest.mark.parametrize(
    ("flag_value", "text_logged"),
    [("true", True), ("false", False)],
)
def test_user_text_logged_only_when_flag_on(
    caplog: pytest.LogCaptureFixture,
    flag_value: str,
    text_logged: bool,
) -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    app.state.database_pool = MagicMock()
    app.state.chatbots = {DEFAULT_MODEL_NAME: MagicMock()}
    env = {"LOG_USER_TEXT": flag_value}

    with (
        patch.dict(os.environ, env),
        patch(
            "api.call_agent",
            return_value={"conversation_id": str(conversation_id), "answer": "ok"},
        ),
        caplog.at_level(logging.INFO, logger="api"),
    ):
        client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "my secret question"},
        )

    assert ("my secret question" in caplog.text) is text_logged


# Checks the flag parsing: blank or "false" is off, "true" is on, anything else is an error.
def test_log_user_text_flag_parsing() -> None:
    with patch.dict(os.environ, {"LOG_USER_TEXT": "TRUE"}):
        assert get_log_user_text() is True
    with patch.dict(os.environ, {}, clear=True):
        assert get_log_user_text() is False
    with patch.dict(os.environ, {"LOG_USER_TEXT": "maybe"}), pytest.raises(RuntimeError):
        get_log_user_text()


# Checks that a divide by zero writes a warning, and the user still gets a friendly answer.
def test_divide_by_zero_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="tools"):
        result = calculator_tool.invoke({"operation": "divide", "num1": 1, "num2": 0})

    assert result == "Cannot divide by zero"
    assert "reason=divide_by_zero" in caplog.text


# Checks that an unknown timezone writes a warning with the bad value.
def test_unknown_timezone_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="tools"):
        get_current_datetime.invoke({"timezone": "Mars/Base"})

    assert "unknown timezone timezone=Mars/Base" in caplog.text


# Checks that the tool error handler logs the error type and returns the safe message.
def test_tool_error_handler_logs_type_and_returns_safe_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="chatbot"):
        message = log_tool_error(ValueError("secret detail"))

    assert message == TOOL_ERROR_MESSAGE
    assert "tool failed error_type=ValueError" in caplog.text
    # The error text itself is not logged, only its type.
    assert "secret detail" not in caplog.text


# Checks that missing settings are logged, then startup stops with the error.
def test_startup_logs_missing_database_url(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("api.configure_logging"),
        patch.dict(os.environ, {}, clear=True),
        caplog.at_level(logging.ERROR, logger="api"),
        pytest.raises(RuntimeError),
        TestClient(app),
    ):
        pass

    assert "startup failed, check environment variables" in caplog.text
