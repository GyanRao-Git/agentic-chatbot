from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from api import app
from exceptions import AgentExecutionError
from tools import CHAT_TOOLS


def use_database_connection(connection: MagicMock) -> MagicMock:
    """Give the FastAPI app a fake database pool for one test."""
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection

    database_pool = MagicMock()
    database_pool.connection.return_value = connection_context
    app.state.database_pool = database_pool

    return database_pool


def test_lifespan_creates_shared_resources() -> None:
    database_pool = MagicMock()
    checkpointer = MagicMock()
    model = MagicMock()
    model_with_tools = MagicMock()
    model.bind_tools.return_value = model_with_tools
    chatbot = MagicMock()

    with (
        patch("api.get_database_url", return_value="postgresql://test"),
        patch("api.get_database_pool_limits", return_value=(2, 5)),
        patch("api.ConnectionPool", return_value=database_pool) as pool_mock,
        patch("api.PostgresSaver", return_value=checkpointer) as checkpointer_mock,
        patch("api.create_chat_model", return_value=model) as model_mock,
        patch("api.build_chatbot", return_value=chatbot) as chatbot_mock,
        TestClient(app) as client,
    ):
        response = client.get("/health")

        assert app.state.database_pool is database_pool
        assert app.state.checkpointer is checkpointer
        assert app.state.chatbot is chatbot

    assert response.status_code == 200
    pool_mock.assert_called_once_with(
        conninfo="postgresql://test",
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=2,
        max_size=5,
        open=False,
    )
    database_pool.__enter__.assert_called_once_with()
    database_pool.__exit__.assert_called_once()
    checkpointer_mock.assert_called_once_with(database_pool)
    model_mock.assert_called_once_with(provider="gemini")
    model.bind_tools.assert_called_once_with(CHAT_TOOLS)
    chatbot_mock.assert_called_once_with(
        model=model_with_tools,
        checkpointer=checkpointer,
        tools=CHAT_TOOLS,
    )


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_conversation_endpoint() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with patch(
        "api.create_conversation",
        return_value=conversation_id,
    ) as create_mock:
        response = client.post("/conversations")

    assert response.status_code == 201
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "title": None,
    }
    create_mock.assert_called_once_with(connection, None)


def test_create_conversation_endpoint_accepts_title() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with patch(
        "api.create_conversation",
        return_value=conversation_id,
    ) as create_mock:
        response = client.post(
            "/conversations",
            json={"title": "  Learning LangGraph  "},
        )

    assert response.status_code == 201
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "title": "Learning LangGraph",
    }
    create_mock.assert_called_once_with(connection, "Learning LangGraph")


def test_list_conversations_endpoint() -> None:
    client = TestClient(app)
    newest_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    older_id = UUID("019435d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with patch(
        "api.list_conversations",
        return_value=[
            (newest_id, "Newest conversation"),
            (older_id, None),
        ],
    ) as list_mock:
        response = client.get("/conversations")

    assert response.status_code == 200
    assert response.json() == [
        {
            "conversation_id": str(newest_id),
            "title": "Newest conversation",
        },
        {
            "conversation_id": str(older_id),
            "title": None,
        },
    ]
    list_mock.assert_called_once_with(
        connection,
        limit=20,
        offset=0,
    )


def test_list_conversations_endpoint_accepts_pagination() -> None:
    client = TestClient(app)
    connection = MagicMock()
    use_database_connection(connection)

    with patch("api.list_conversations", return_value=[]) as list_mock:
        response = client.get("/conversations?limit=5&offset=10")

    assert response.status_code == 200
    assert response.json() == []
    list_mock.assert_called_once_with(
        connection,
        limit=5,
        offset=10,
    )


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
    ],
)
def test_list_conversations_endpoint_rejects_invalid_pagination(query: str) -> None:
    client = TestClient(app)

    response = client.get(f"/conversations?{query}")

    assert response.status_code == 422


def test_rename_conversation_endpoint() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with (
        patch("api.conversation_exists", return_value=True),
        patch("api.update_conversation_title") as update_mock,
    ):
        response = client.patch(
            f"/conversations/{conversation_id}",
            json={"title": "  Renamed conversation  "},
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "title": "Renamed conversation",
    }
    update_mock.assert_called_once_with(
        connection=connection,
        conversation_id=conversation_id,
        title="Renamed conversation",
    )


def test_rename_conversation_endpoint_returns_not_found() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with (
        patch("api.conversation_exists", return_value=False),
        patch("api.update_conversation_title") as update_mock,
    ):
        response = client.patch(
            f"/conversations/{conversation_id}",
            json={"title": "Unknown conversation"},
        )

    assert response.status_code == 404
    update_mock.assert_not_called()


def test_conversation_title_rejects_blank_value() -> None:
    client = TestClient(app)

    response = client.post(
        "/conversations",
        json={"title": "   "},
    )

    assert response.status_code == 422


def test_conversation_title_rejects_more_than_100_characters() -> None:
    client = TestClient(app)

    response = client.post(
        "/conversations",
        json={"title": "a" * 101},
    )

    assert response.status_code == 422


def test_delete_conversation_endpoint() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    checkpointer = MagicMock()
    app.state.checkpointer = checkpointer

    with (
        patch("api.conversation_exists", return_value=True),
        patch("api.delete_conversation_history") as history_mock,
        patch("api.delete_conversation") as delete_mock,
    ):
        response = client.delete(f"/conversations/{conversation_id}")

    assert response.status_code == 204
    assert response.content == b""
    history_mock.assert_called_once_with(
        conversation_id=conversation_id,
        checkpointer=checkpointer,
    )
    delete_mock.assert_called_once_with(connection, conversation_id)


def test_delete_conversation_endpoint_returns_not_found() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)

    with (
        patch("api.conversation_exists", return_value=False),
        patch("api.delete_conversation_history") as history_mock,
        patch("api.delete_conversation") as delete_mock,
    ):
        response = client.delete(f"/conversations/{conversation_id}")

    assert response.status_code == 404
    history_mock.assert_not_called()
    delete_mock.assert_not_called()


def test_get_messages_endpoint() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    checkpointer = MagicMock()
    app.state.checkpointer = checkpointer

    with patch(
        "api.get_conversation_messages",
        return_value=(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi Gyan"},
            ],
            None,
        ),
    ) as messages_mock:
        response = client.get(f"/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi Gyan"},
        ],
        "next_before": None,
    }
    messages_mock.assert_called_once_with(
        conversation_id=conversation_id,
        database_connection=connection,
        checkpointer=checkpointer,
        debug=False,
        limit=50,
        before=None,
    )


def test_get_messages_endpoint_includes_tool_output_in_debug_mode() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    checkpointer = MagicMock()
    app.state.checkpointer = checkpointer

    with patch(
        "api.get_conversation_messages",
        return_value=(
            [
                {"role": "tool", "content": "Current date and time in UTC: ..."},
            ],
            None,
        ),
    ) as messages_mock:
        response = client.get(
            f"/conversations/{conversation_id}/messages?debug=true"
        )

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "tool", "content": "Current date and time in UTC: ..."}
    ]
    messages_mock.assert_called_once_with(
        conversation_id=conversation_id,
        database_connection=connection,
        checkpointer=checkpointer,
        debug=True,
        limit=50,
        before=None,
    )


def test_get_messages_endpoint_accepts_pagination_cursor() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    checkpointer = MagicMock()
    app.state.checkpointer = checkpointer

    with patch(
        "api.get_conversation_messages",
        return_value=(
            [{"role": "assistant", "content": "Older answer"}],
            4,
        ),
    ) as messages_mock:
        response = client.get(
            f"/conversations/{conversation_id}/messages"
            "?debug=true&limit=2&before=6"
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "messages": [{"role": "assistant", "content": "Older answer"}],
        "next_before": 4,
    }
    messages_mock.assert_called_once_with(
        conversation_id=conversation_id,
        database_connection=connection,
        checkpointer=checkpointer,
        debug=True,
        limit=2,
        before=6,
    )


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "before=-1",
    ],
)
def test_get_messages_endpoint_rejects_invalid_pagination(query: str) -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")

    response = client.get(
        f"/conversations/{conversation_id}/messages?{query}"
    )

    assert response.status_code == 422


def test_get_messages_endpoint_returns_not_found() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    app.state.checkpointer = MagicMock()

    with patch(
        "api.get_conversation_messages",
        side_effect=ValueError(f"Conversation does not exist: {conversation_id}"),
    ):
        response = client.get(f"/conversations/{conversation_id}/messages")

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Conversation does not exist: {conversation_id}"
    }


def test_chat_endpoint_returns_structured_agent_response() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    chatbot = MagicMock()
    app.state.chatbot = chatbot

    with patch(
        "api.call_agent",
        return_value={
            "conversation_id": str(conversation_id),
            "answer": "Hello from the agent",
        },
    ) as send_mock:
        response = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "Hello"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "answer": "Hello from the agent",
    }
    send_mock.assert_called_once_with(
        conversation_id=conversation_id,
        user_input="Hello",
        database_pool=app.state.database_pool,
        chatbot=chatbot,
    )


def test_chat_endpoint_returns_not_found_for_unknown_conversation() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    app.state.chatbot = MagicMock()

    with patch(
        "api.call_agent",
        side_effect=ValueError(f"Conversation does not exist: {conversation_id}"),
    ) as send_mock:
        response = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "Hello"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Conversation does not exist: {conversation_id}"
    }
    send_mock.assert_called_once_with(
        conversation_id=conversation_id,
        user_input="Hello",
        database_pool=app.state.database_pool,
        chatbot=app.state.chatbot,
    )


def test_chat_endpoint_returns_safe_error_when_agent_fails() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = MagicMock()
    use_database_connection(connection)
    app.state.chatbot = MagicMock()

    with patch(
        "api.call_agent",
        side_effect=AgentExecutionError(),
    ):
        response = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "Hello"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The agent could not complete the request. Please try again."
    }


def test_chat_endpoint_rejects_blank_message() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")

    response = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"message": "   "},
    )

    assert response.status_code == 422
