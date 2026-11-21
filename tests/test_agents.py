import logging
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage

from agents import call_agent
from exceptions import AgentExecutionError


def make_database_pool() -> MagicMock:
    """Return a fake pool whose connection() context manager lends a fake connection."""
    database_pool = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = MagicMock()
    database_pool.connection.return_value = connection_context
    return database_pool


def test_call_agent_reuses_shared_chatbot() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_pool = make_database_pool()
    connection = database_pool.connection.return_value.__enter__.return_value
    chatbot = Mock()
    chatbot.invoke.return_value = {
        "messages": [AIMessage(content="Hello from Gemini")]
    }

    with patch("agents.conversation_exists", return_value=True) as exists_mock:
        result = call_agent(
            conversation_id=conversation_id,
            user_input="Hello",
            database_pool=database_pool,
            chatbot=chatbot,
        )

    assert result == {
        "conversation_id": str(conversation_id),
        "answer": "Hello from Gemini",
    }
    database_pool.connection.assert_called_once_with()
    exists_mock.assert_called_once_with(
        connection=connection,
        conversation_id=conversation_id,
    )

    state = chatbot.invoke.call_args.args[0]
    assert state["messages"][0].content == "Hello"
    assert chatbot.invoke.call_args.kwargs["config"] == {
        "configurable": {"thread_id": str(conversation_id)}
    }


def test_call_agent_returns_connection_before_calling_model() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_pool = make_database_pool()
    connection_context = database_pool.connection.return_value

    # Record the order of events: the connection is released first,
    # then the model is called.
    events: list[str] = []
    connection_context.__exit__.side_effect = lambda *args: events.append("released")

    def fake_invoke(state: object, config: object) -> dict[str, list[AIMessage]]:
        events.append("model called")
        return {"messages": [AIMessage(content="Hi")]}

    chatbot = Mock()
    chatbot.invoke.side_effect = fake_invoke

    with patch("agents.conversation_exists", return_value=True):
        call_agent(
            conversation_id=conversation_id,
            user_input="Hello",
            database_pool=database_pool,
            chatbot=chatbot,
        )

    assert events == ["released", "model called"]


def test_call_agent_rejects_unknown_conversation() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_pool = make_database_pool()
    chatbot = Mock()

    with (
        patch("agents.conversation_exists", return_value=False),
        pytest.raises(ValueError, match="Conversation does not exist"),
    ):
        call_agent(
            conversation_id=conversation_id,
            user_input="Hello",
            database_pool=database_pool,
            chatbot=chatbot,
        )

    chatbot.invoke.assert_not_called()


def test_call_agent_wraps_and_safely_logs_graph_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_pool = make_database_pool()
    chatbot = Mock()
    chatbot.invoke.side_effect = RuntimeError("private provider response")

    with (
        patch("agents.conversation_exists", return_value=True),
        caplog.at_level(logging.ERROR, logger="agents"),
        pytest.raises(AgentExecutionError) as error_info,
    ):
        call_agent(
            conversation_id=conversation_id,
            user_input="private user message",
            database_pool=database_pool,
            chatbot=chatbot,
        )

    assert str(error_info.value) == (
        "The agent could not complete the request. Please try again."
    )
    assert str(conversation_id) in caplog.text
    assert "RuntimeError" in caplog.text
    assert "private provider response" not in caplog.text
    assert "private user message" not in caplog.text
