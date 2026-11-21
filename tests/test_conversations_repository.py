from unittest.mock import Mock
from uuid import UUID

import pytest

from conversations_repository import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    list_conversations,
    update_conversation_title,
)


def test_create_conversation_returns_database_generated_uuid() -> None:
    expected_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = {"id": expected_id}

    result = create_conversation(connection, "Learning LangGraph")

    assert result == expected_id
    query, parameters = connection.execute.call_args.args
    assert "INSERT INTO conversations (title)" in query
    assert parameters == ("Learning LangGraph",)


def test_create_conversation_raises_when_database_returns_no_row() -> None:
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = None

    with pytest.raises(RuntimeError, match="Could not create a row"):
        create_conversation(connection)


def test_list_conversations_returns_database_ids() -> None:
    newest_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    older_id = UUID("019435d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()
    connection.execute.return_value.fetchall.return_value = [
        {"id": newest_id, "title": "Newest conversation"},
        {"id": older_id, "title": None},
    ]

    result = list_conversations(connection)

    assert result == [
        (newest_id, "Newest conversation"),
        (older_id, None),
    ]
    query, parameters = connection.execute.call_args.args
    assert "ORDER BY id DESC" in query
    assert "LIMIT %s" in query
    assert "OFFSET %s" in query
    assert parameters == (20, 0)


def test_list_conversations_applies_requested_pagination() -> None:
    connection = Mock()
    connection.execute.return_value.fetchall.return_value = []

    result = list_conversations(
        connection,
        limit=5,
        offset=10,
    )

    assert result == []
    query, parameters = connection.execute.call_args.args
    assert "LIMIT %s" in query
    assert "OFFSET %s" in query
    assert parameters == (5, 10)


def test_update_conversation_title_uses_conversation_id() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()

    update_conversation_title(
        connection=connection,
        conversation_id=conversation_id,
        title="Renamed conversation",
    )

    query, parameters = connection.execute.call_args.args
    assert "UPDATE conversations" in query
    assert parameters == ("Renamed conversation", conversation_id)


def test_delete_conversation_uses_conversation_id() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()

    delete_conversation(connection, conversation_id)

    connection.execute.assert_called_once_with(
        "DELETE FROM conversations WHERE id = %s",
        (conversation_id,),
    )


@pytest.mark.parametrize(("database_value", "expected"), [(True, True), (False, False)])
def test_conversation_exists_returns_database_result(
    database_value: bool,
    expected: bool,
) -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = {
        "conversation_exists": database_value
    }

    result = conversation_exists(connection, conversation_id)

    assert result is expected
    query, parameters = connection.execute.call_args.args
    assert "SELECT EXISTS" in query
    assert parameters == (conversation_id,)


def test_conversation_exists_raises_when_database_returns_no_row() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = None

    with pytest.raises(RuntimeError, match="Could not check conversation_id"):
        conversation_exists(connection, conversation_id)
